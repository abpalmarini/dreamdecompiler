# @DreamDecompiler

import itertools
import math
import gc
from collections import Counter, namedtuple
from operator import itemgetter

from dreamcoder.fragmentUtilities import (RewriteFragments, canonicalFragment, defragment,
                                          proposeFragmentsFromFrontiers, fragmentSize)
from dreamcoder.program import Application, PlaceholderIndex, Index, InferenceFailure
from dreamcoder.type import TypeVariable, Context, canUnify
from dreamcoder.utilities import eprint, lse, NEGATIVEINFINITY, parallelMap, timing
from dreamcoder.grammar import Grammar, LikelihoodSummary, ContextualGrammar
from dreamcoder.vs import VersionTable
from dreamcoder.frontier import Frontier


class DreamDecompiler:
    """
    useIndexedTypes: Whether indexed types should be used when averaging over
    likelihoods of polymorphic fragment requests (the number of mappings
    increases exponentially with types, but most mappings using indexed types
    will have the same likelihood as those without so the default is False).

    useProgramPrimArgTypeCounts: Whether the probability of parent nodes, argument
    index and argument types should be proporional to their frequencies in programs
    or have a unifrom distribution based on all primitive productions when
    calculating the marginal likelihood of a fragment.
    """

    def __init__(self, 
        recognitionModel,
        allFrontiers,
        useIndexedTypes=False,
        useProgramPrimArgTypeCounts=True
    ):
        self.recognitionModel = recognitionModel
        self.contextual = recognitionModel.contextual
        self.generativeModel = recognitionModel.generativeModel
        self.grammar = recognitionModel.grammar  # used for LS calculation only (not LL)
        self.librarySize = 2 + sum(len(t.functionArguments()) for _, t, _ in self.grammar.productions)
        self.allFrontiers = allFrontiers 
        self.tasks = [frontier.task for frontier in allFrontiers]
        self.taskGrammars = {task: recognitionModel.grammarOfTask(task).untorch() for task in self.tasks}
        baseTypes, allTypes = self.typesUsed()
        self.typesForMapping = allTypes if useIndexedTypes else baseTypes
        self.useProgramPrimArgTypeCounts = useProgramPrimArgTypeCounts
        if useProgramPrimArgTypeCounts:
            self.primArgTypeCounts = self.findPrimArgTypeCounts()
            self.totalPrimArgTypes = sum(count for count in self.primArgTypeCounts.values())

        # Caches 
        self.variableTypeMappingsCache = {}  
        self.unifyingLibraryPrimArgsCache = {}
        self.unifyingProgramPrimArgsCache = {}

    def typesUsed(self):
        """
        Returns the set of all base and indexed (depth of one) types used by the 
        primitives in the generative model's grammar.
        """
        # First pass finds all base types
        baseTypes = set()
        for _, t, _ in self.generativeModel.productions:
            returnType = t.returns()
            if not isinstance(returnType, TypeVariable) and not returnType.arguments:
                baseTypes.add(returnType)
        
        # Second pass finds the indexed types (lists, pairs, etc.) as well
        allTypes = set()
        for _, t, _ in self.generativeModel.productions:
            returnType = t.returns()
            if not returnType.isPolymorphic:
                allTypes.add(returnType)
            elif not isinstance(returnType, TypeVariable):
                mappings = self.variableTypeMappings(returnType, baseTypes)
                for mapping in mappings:
                    allTypes.add(returnType.makeDummyMonomorphic(mapping))

        return baseTypes, allTypes

    def variableTypeMappings(self, pType, typesForMapping=None):
        """
        Returns list of all possible mappings that the free type variables in
        the given program type could refer to from a set of types.
        """
        assert pType.isPolymorphic, f"{pType} has no type variables."

        variables = tuple(sorted(pType.freeTypeVariables()))
        cache = typesForMapping is None
        if cache:
            if variables in self.variableTypeMappingsCache:
                return self.variableTypeMappingsCache[variables]
            typesForMapping = self.typesForMapping

        mappings = []
        for mapping in itertools.product(typesForMapping, repeat=len(variables)):
            mappings.append({v: t for v, t in zip(variables, mapping)})

        if cache:
            self.variableTypeMappingsCache[variables] = mappings

        return mappings

    def unifyingLibraryPrimArgs(self, request):
        """
        Return all (primitive, arg) pairs from all library productions whose type
        unifies with the given request.
        """
        if request in self.unifyingLibraryPrimArgsCache:
            return self.unifyingLibraryPrimArgsCache[request]
        primArgs = []
        for _, primType, prim in self.generativeModel.productions:
            for arg, argType in enumerate(primType.functionArguments()):
                # If implication type we only care about the final return type.
                if canUnify(request, argType.returns()):
                    primArgs.append((prim, arg))
        self.unifyingLibraryPrimArgsCache[request] = primArgs
        return primArgs

    def findPrimArgTypeCounts(self):
        """
        Counts up all parent node, parent arg, type request triplets for each node
        of each program in the frontier.
        """
        allPrimArgTypes = []

        # Similar structure to Grammar's likelihood summary calculation.
        def addProgramPrimArgTypes(
            parent,
            parentIndex,
            context,
            environment,
            request,
            expression
        ):
            if request.isArrow():
                return addProgramPrimArgTypes(
                    parent,
                    parentIndex,
                    context, 
                    [request.arguments[0]] + environment,
                    request.arguments[1],
                    expression.body
                )
            # Store canonical type to merge those that differ only in variable number.
            allPrimArgTypes.append((parent, parentIndex, request.canonical()))

            candidates = self.generativeModel.buildCandidates(
                request,
                context,
                environment,
                normalize=False,
                returnTable=True
            )
            f, xs = expression.applicationParse()
            _, t, context = candidates[f]
            argTypes = t.functionArguments()
            for argIndex, (argType, arg) in enumerate(zip(argTypes, xs)):
                argType = argType.apply(context) 
                context = addProgramPrimArgTypes(f, argIndex, context, environment, argType, arg)

            return context

        for f in self.allFrontiers:
            request = f.task.request
            for e in f.entries:
                program = e.program
                addProgramPrimArgTypes(None, None, Context.EMPTY, [], request, program)
        
        return Counter(allPrimArgTypes)

    def unifyingProgramPrimArgs(self, request):
        """
        Return all primitive, args and their respective count for each primitive,
        arg, type found in our programs that unifies with the given request.
        """
        if request in self.unifyingProgramPrimArgsCache:
            return self.unifyingProgramPrimArgsCache[request]
        primArgs = []
        for (prim, arg, argType), count in self.primArgTypeCounts.items():
            if canUnify(request, argType):
                primArgs.append((prim, arg, count))
        self.unifyingProgramPrimArgsCache[request] = primArgs
        return primArgs

    # Grammar's define a probability distribution over programs of a requested
    # type so to get the probability of a Grammar generating a fragment we must
    # ensure it is a self contained, well typed program. We do this by adding in
    # placeholder indices that won't be considered in calculating the likelihood
    # of the program: thus, giving us the likelihood of the fragment.
    def fragmentToProgram(self, fragment):
        """
        Augment a fragment with necessary placeholders so that it becomes a complete,
        well typed program with no free variables.
        """
        # Ensure all free variables are bound
        p = canonicalFragment(fragment)
        p = p.wrapInAbstractions(p.numberOfFreeVariables)

        # Perform eta expansion for each expression with an
        # implication type that is not an abstraction:
        # M of type a -> b becomes 𝜆x. N x of type a -> b
        # (Same as uncurry but with placeholders.)
        pType = p.infer()
        numArgs = len(pType.functionArguments())
        numAbstractions = 0
        while p.isAbstraction:
            p = p.body
            numAbstractions += 1
        neededAbstractions = numArgs - numAbstractions
        p = p.shift(neededAbstractions)
        for i in reversed(range(neededAbstractions)):
            p = Application(p, PlaceholderIndex(i))
        p = p.wrapInAbstractions(numArgs)

        return p, pType

    def fragmentLikelihoodSummary(self, fragment):
        """
        Returns the likelihood summary and return type of a given fragment.
        """
        p, pType = self.fragmentToProgram(fragment)
        if pType.isPolymorphic:
            # Negate type variables so they don't clash with type variables
            # in primitive types when choosing candidates for program nodes.
            pType = pType.negateVariables()
        try:
            ls = self.grammar.closedLikelihoodSummary(pType, p)
            return ls, pType.returns()
        except AssertionError:
            return None, pType.returns()

    def allMonomorphicfragmentSummaries(self, fragment):
        """
        Returns a list of likelihood summaries (and return types) for different
        monomorphic mappings of the type variables in a fragment type. (Different
        types can mean different candidates and thus likelihoods.)
        """
        likelihoodSummaries = []
        p, pType = self.fragmentToProgram(fragment)
        if pType.isPolymorphic:
            for mapping in self.variableTypeMappings(pType):
                t = pType.makeDummyMonomorphic(mapping)
                try:
                    ls = self.grammar.closedLikelihoodSummary(t, p)
                    likelihoodSummaries.append((ls, t.returns()))
                except AssertionError:
                    continue
        else:
            try:
                ls = self.grammar.closedLikelihoodSummary(pType, p)
                likelihoodSummaries.append((ls, pType.returns()))
            except AssertionError:
                pass

        return likelihoodSummaries 

    def marginalizedLL(self, fragmentLS, returnType, taskGrammar):
        """
        Calculates the log likelihood of the recognition model generating a fragment
        (given its likelihood summary and return type) on a specifc task grammar by
        marginalizing over parent nodes, argument index and argument types that could
        validly generate the fragment. For unigram models only argument types are
        marginalized as the productions do not depend on parents.
        """
        if self.contextual:
            headLS = fragmentLS.noParent
            fragmentLS.noParent = LikelihoodSummary()  # now LS for fragment body only

            bodyLL = fragmentLS.logLikelihood(taskGrammar)

            if self.useProgramPrimArgTypeCounts:
                # Marginalise parent-arg-types generating head with distribution proportionate
                # to their frequency in programs.
                headL = 0
                for primitive, arg, count in self.unifyingProgramPrimArgs(returnType):
                    if primitive is None:
                        parentArgGrammar = taskGrammar.noParent
                    elif primitive.isIndex:
                        parentArgGrammar = taskGrammar.variableParent
                    else:
                        parentArgGrammar = taskGrammar.library[primitive][arg]
                    headL += math.exp(headLS.logLikelihood(parentArgGrammar)) * count
                headLL = - math.log(self.totalPrimArgTypes) + math.log(headL)
            else:
                # Marginalise parent-arg-types generating head with a unifrom distribution
                # over all primitive productions.
                headLLs = []
                headLLs.append(headLS.logLikelihood(taskGrammar.noParent))
                headLLs.append(headLS.logLikelihood(taskGrammar.variableParent))
                for primitive, arg in self.unifyingLibraryPrimArgs(returnType):
                    parentArgGrammar = taskGrammar.library[primitive][arg]
                    headLLs.append(headLS.logLikelihood(parentArgGrammar))
                headLL = - math.log(self.librarySize) + lse(headLLs)

            return bodyLL + headLL
        else:
            # For a unigram model, generating the head is independent of the parent and arg
            # but the likelihood of the fragment being generated still depends on occurrences
            # of nodes requesting a certain type.
            LL = fragmentLS.logLikelihood(taskGrammar)
            if self.useProgramPrimArgTypeCounts:
                typeCounts = sum(count for _, _, count in self.unifyingProgramPrimArgs(returnType))
                return LL + math.log(typeCounts) - math.log(self.totalPrimArgTypes)
            else:
                typeCounts = len(self.unifyingLibraryPrimArgs(returnType)) + 2  # 2 for no parent and variable
                return LL + math.log(typeCounts) - math.log(self.librarySize)

    def expectedLLOverTasks(self, fragment, fromRoot):
        """
        Returns the expected log likelihood of the recognition model generating
        a fragment over all tasks. If fromRoot is ture then the likilhood is that
        of generating the fragment as a standalone progam. Otherwise, the parent nodes,
        argument index and argument type requests that the fragment could be generated
        from are also marginalised over.
        """
        ls, returnType = self.fragmentLikelihoodSummary(fragment)
        if ls is None:
            p, pType = self.fragmentToProgram(fragment)
            eprint(f"Fragment {fragment} has no likelihood summary. ({p} : {pType})")
            return NEGATIVEINFINITY

        logLikelihoods = []
        if fromRoot:
            for grammar in self.taskGrammars.values():
                logLikelihoods.append(ls.logLikelihood(grammar))
        else:
            for grammar in self.taskGrammars.values():
                logLikelihoods.append(self.marginalizedLL(ls, returnType, grammar))

        return - math.log(len(self.taskGrammars)) + lse(logLikelihoods)

    def programContainsFragment(self, program, fragment):
        """
        Returns true if the given fragment is a node anywhere in the given program.
        """
        fragment = defragment(fragment)  # Ensure fragment is same form as would be used in programs.
        for _, p in program.walk():
            if p == fragment:
                return True
        return False

    # We can't calculate likelihood summary like normal and remove uses of the fragment
    # because the fragment is not part of the grammar yet. We return the requests so the
    # actual likelihood of generating all fragment occurrences in the program can be 
    # calculated. This is needed because fragments that have polymorphic return types can
    # have different likelihood summaries depending on the monomorphic type used. 
    def unigramRewriteSummaryInfo(self, concreteFragment, program, programRequest):
        """
        Returns the likelihood summary of generating all parts of the given program
        other than the concrete fragment as well as the type requests for each occurrence
        of the fragment in the program. 
        """
        assert not self.contextual

        fragmentRequests = []

        # Need a dummy grammar containing fragment for identification and context details.
        dummyGrammar = Grammar.uniform(self.grammar.primitives + [concreteFragment])

        # Similar to Grammar's likelihood summary calculation except a dummy grammar is used
        # and the fragment is neither stored in summary or normalisers.
        def nonFragmentLS(context, environment, request, expression):
            if request.isArrow():
                if not expression.isAbstraction:
                    eprint(f"Request is an arrow but I got {expression} when finding non fragment LS.")
                    return context, None
                return nonFragmentLS(
                    context,
                    [request.arguments[0]] + environment,
                    request.arguments[1],
                    expression.body
                )
            candidates = dummyGrammar.buildCandidates(
                request,
                context,
                environment,
                normalize=False,
                returnTable=True
            )
            # Don't include the abstracted fragment in normalisers.
            possibles = [p for p in candidates.keys() if not (p.isIndex or p == concreteFragment)]
            numberOfVariables = sum(p.isIndex for p in candidates.keys())
            if numberOfVariables > 0:
                possibles += [Index(0)]

            f, xs = expression.applicationParse()
            if f not in candidates:
                eprint(f, "not in candidates for request", request)
                return context, None

            thisSummary = LikelihoodSummary()
            if f == concreteFragment:
                fragmentRequests.append(request)
            else:
                constant = -math.log(numberOfVariables) if f.isIndex else 0
                thisSummary.record(f, possibles, constant=constant)

            _, tp, context = candidates[f]
            argumentTypes = tp.functionArguments()
            for argumentType, argument in zip(argumentTypes, xs):
                argumentType = argumentType.apply(context)
                context, newSummary = nonFragmentLS(context, environment, argumentType, argument)
                if newSummary is None:
                    return context, None
                thisSummary.join(newSummary)
            
            return context, thisSummary

        _, programLS = nonFragmentLS(Context.EMPTY, [], programRequest, program)
        return programLS, fragmentRequests

    def contextualRewriteSummaryInfo(self, concreteFragment, program, programRequest):
        """
        Contextual version of `unigramRewriteSummaryInfo`. Returns the likelihood
        summary of generating all parts of the given program other than the concrete fragment,
        a list of likelihood summaries for the immediate children of any uses of the fragment
        in the rewritten program (as it is not part of the grammar) and a list containg the
        details of how all occurrences of the fragment appeared: the parent and parent index
        generating the fragment as well as the type request.
        """
        assert self.contextual

        fragmentOccurrences = []

        # Need a dummy grammar containing fragment for identification and context details.
        dummyGrammar = Grammar.uniform(self.grammar.primitives + [concreteFragment])

        # A contextual model takes parents into account, but our fragment isn't part of
        # our grammar so we need to create extra summaries to track immediate children uses.
        # (Later, when calculating the probability of generating a rewritten program, a
        # uniform grammar will be used for these immediate fragment children.)
        numArgs = len(concreteFragment.infer().functionArguments())
        fragmentAsParentLSs = [LikelihoodSummary() for _ in range(numArgs)]

        def nonFragmentLS(parent, parentIndex, context, environment, request, expression):
            if request.isArrow():
                if not expression.isAbstraction:
                    eprint(f"Request is an arrow but I got {expression} when finding non fragment LS.")
                    return context, None
                return nonFragmentLS(
                    parent,
                    parentIndex,
                    context,
                    [request.arguments[0]] + environment,
                    request.arguments[1],
                    expression.body
                )
            candidates = dummyGrammar.buildCandidates(
                request,
                context,
                environment,
                normalize=False,
                returnTable=True
            )
            # Don't include the abstracted fragment in normalisers.
            possibles = [p for p in candidates.keys() if not (p.isIndex or p == concreteFragment)]
            numberOfVariables = sum(p.isIndex for p in candidates.keys())
            if numberOfVariables > 0:
                possibles += [Index(0)]

            f, xs = expression.applicationParse()
            if f not in candidates:
                eprint(f, "not in candidates for request", request)
                return context, None

            thisSummary = ContextualGrammar.LS(self.grammar)
            if f == concreteFragment:
                fragmentOccurrences.append((parent, parentIndex, request))
            else:
                constant= -math.log(numberOfVariables) if f.isIndex else 0
                if parent == concreteFragment:
                    # Record immediate children of fragment separately.
                    fragmentAsParentLSs[parentIndex].record(f, possibles, constant=constant)
                else:
                    thisSummary.record(parent, parentIndex, f, possibles, constant=constant)

            _, tp, context = candidates[f]
            argumentTypes = tp.functionArguments()
            for i, (argumentType, argument) in enumerate(zip(argumentTypes, xs)):
                argumentType = argumentType.apply(context)
                context, newSummary = nonFragmentLS(f, i, context, environment, argumentType, argument)
                if newSummary is None:
                    return context, None
                thisSummary.join(newSummary)
            
            return context, thisSummary

        _, programLS = nonFragmentLS(None, None, Context.EMPTY, [], programRequest, program)
        return programLS, fragmentAsParentLSs, fragmentOccurrences

    def fragmentSummariesForRequests(self, fragment, requests, fragmentSummaries=None):
        """
        Return a list of likelihood summaries of the given fragment for each
        return type request.
        """
        fragmentSummaries = fragmentSummaries if fragmentSummaries is not None else {}
        p, pType = self.fragmentToProgram(fragment)
        returnType = pType.returns()

        # Only one summary possible if return type is monomorphic.
        if not returnType.isPolymorphic:
            if returnType not in fragmentSummaries:
                ls, _ = self.fragmentLikelihoodSummary(fragment)
                fragmentSummaries[returnType] = ls
            return [fragmentSummaries[returnType]] * len(requests)

        fragmentLSs = []
        for request in requests:
            # Ensure that no type variables clash which could change the meaning of the
            # fragment type after making a substitution. 
            requestFreeVariables = request.freeTypeVariables()
            if returnType.freeTypeVariables().intersection(requestFreeVariables):
                # Can't negate variables for this or they will be flipped back in pTypeSub
                bindings = {v: TypeVariable(v + 10) for v in requestFreeVariables}
                request = request.canonical(bindings)

            if request in fragmentSummaries:
                fragmentLSs.append(fragmentSummaries[request])
            else:
                # Unify request and return type, then apply the type variable substitutions
                # from the return type to the rest of the program's type.
                k = Context.EMPTY.unify(returnType, request)
                bindings = {v: t for v, t in k.substitution}
                pTypeSub = pType.canonical(bindings) if bindings else pType
                ls = self.grammar.closedLikelihoodSummary(pTypeSub.negateVariables(), p)
                fragmentSummaries[request] = ls
                fragmentLSs.append(ls)
        return fragmentLSs

    def fragmentOccurrencesLL(self, fragment, fragmentOccurrences, grammar, fragmentSummaries):
        """
        Returns the total log likelihood of the given contextual grammar generating each
        occurrence of a fragment where an occurrence is specified by a parent, parent index
        and requested return type generating the fragment.
        """
        concreteFragment = defragment(fragment)
        fragmentRequests = [t for _, _, t in fragmentOccurrences]
        fragmentLSs = self.fragmentSummariesForRequests(fragment, fragmentRequests, fragmentSummaries)
        allFragmentsLL = 0
        for ls, (parent, parentIndex, _) in zip(fragmentLSs, fragmentOccurrences):
            headLS = ls.noParent
            ls.noParent = LikelihoodSummary()  # now LS for fragment body only

            bodyLL = ls.logLikelihood(grammar)

            # Get log likelihood of head with the grammar of the parent (and arg)
            # of the fragment occurrence.
            if parent is None:
                parentGrammar = grammar.noParent
            elif parent.isIndex:
                parentGrammar = grammar.variableParent
            elif parent == concreteFragment:
                # If our fragment is used recursively as a child of itself then we
                # use a uniform grammar as with all immediate fragment children
                parentGrammar = Grammar.uniform(self.grammar.primitives)
            else:
                parentGrammar = grammar.library[parent][parentIndex]
            headLL = headLS.logLikelihood(parentGrammar)

            allFragmentsLL += bodyLL + headLL

            # Replace head to leave ls unmodified as it may be used elsewhere
            ls.noParent = headLS

        return allFragmentsLL

    def chunkGivenTaskProbability(self, fragment, frontier, chunkWeighting, fragmentSummaries=None):
        """
        Returns the probability of chunking a fragment on a certain task (P(c|f, Q, x))
        by marginalising over the programs in the task's frontier.

        chunkWeighting: Whether the probability of chunking a fragment for a given task
        should be weighted by the raw probabilities of the recognition model generating
        each program in the beam ("raw") and thus a lower bound to true marginalisation
        over all programs, just propotionate to the recognition model beam probabilities
        ("prop") or if a unifrom distribution over the programs in the beam should be
        used ("uniform").
        """
        assert chunkWeighting in {"raw", "prop", "uniform"}
        fragmentSummaries = fragmentSummaries if fragmentSummaries is not None else {}
        grammar = self.taskGrammars[frontier.task]
        request = frontier.task.request
        # Rewrite programs in frontier to use fragment if possible before marginalising.
        rewrittenFrontier = RewriteFragments.rewriteFrontier(frontier, fragment)
        concreteFragment = defragment(fragment)

        chunkGivenTaskPr = 0
        normaliser = 0
        for e in rewrittenFrontier.entries:
            if not self.programContainsFragment(e.program, concreteFragment):
                continue  # Probability of chunking on program is 0.

            if e.program == concreteFragment:
                # If rewritten program is equal to fragment then chunking given program probability
                # is 1 and probability of program is just the probability of the fragment.
                if chunkWeighting == "uniform":
                    chunkGivenTaskPr += 1
                else:
                    ls = self.fragmentSummariesForRequests(fragment, [request.returns()], fragmentSummaries)[0]
                    programL = math.exp(ls.logLikelihood(grammar))
                    chunkGivenTaskPr += programL
                    normaliser += programL
                continue

            # Calculate LL of full program and fragment uses part of the program (split for
            # dealing with the contextual or unigram case).
            if self.contextual:
                rewriteSummaryInfo = self.contextualRewriteSummaryInfo(
                    concreteFragment,
                    e.program,
                    request
                )
                programMinusFragmentLS, fragmentAsParentLSs, fragmentOccurrences = rewriteSummaryInfo
                if programMinusFragmentLS is None:
                    # Program has no likelihood summary (occurrs for programs with
                    # indices expecting abstractions) so we give 0 probability.
                    continue  
                allFragmentsLL = self.fragmentOccurrencesLL(
                    fragment,
                    fragmentOccurrences,
                    grammar,
                    fragmentSummaries
                )
                # To get LL of full program we need to sum LL of fragment parts, non fragment parts
                # and the LL of generating each immediate child of the fragment in the program. Given
                # their is no parent grammar for our fragment we use a uniform distribution over the
                # available nodes.
                programMinusFragmentLL = programMinusFragmentLS.logLikelihood(grammar) 
                uniformGrammar = Grammar.uniform(self.grammar.primitives)
                fragmentChildrenLL = sum(ls.logLikelihood(uniformGrammar) for ls in fragmentAsParentLSs)
                programLL = allFragmentsLL + programMinusFragmentLL + fragmentChildrenLL
            else:
                programMinusFragmentLS, fragmentRequests = self.unigramRewriteSummaryInfo(
                    concreteFragment,
                    e.program,
                    request
                )
                if programMinusFragmentLS is None:
                    continue  
                fragmentLSs = self.fragmentSummariesForRequests(
                    fragment,
                    fragmentRequests,
                    fragmentSummaries
                )
                allFragmentsLL = sum(ls.logLikelihood(grammar) for ls in fragmentLSs)
                programLL = allFragmentsLL + programMinusFragmentLS.logLikelihood(grammar)

            if chunkWeighting == "uniform":
                chunkGivenTaskPr += (allFragmentsLL / programLL)
            else:
                programL = math.exp(programLL)
                chunkGivenTaskPr += (allFragmentsLL / programLL) *  programL
                normaliser += programL
        
        if chunkWeighting == "uniform":
            chunkGivenTaskPr /= len(rewrittenFrontier)
        elif chunkWeighting == "prop" and chunkGivenTaskPr != 0:
            chunkGivenTaskPr /= normaliser

        return chunkGivenTaskPr

    def fragmentChunkProbability(self, fragment, fromRoot, chunkWeighting):
        """
        Calculates the probability (belief) of how useful chunking a given fragment
        will be for the recognition model. 

        chunkWeighting: Whether the probability of chunking a fragment for a given task
        should be weighted by the raw probabilities of the recognition model generating
        each program in the beam ("raw") and thus a lower bound to true marginalisation
        over all programs, just propotionate to the recognition model beam probabilities
        ("prop") or if a unifrom distribution over the programs in the beam should be
        used ("uniform").
        """
        ls, returnType = self.fragmentLikelihoodSummary(fragment)
        if ls is None:
            p, pType = self.fragmentToProgram(fragment)
            eprint(f"Fragment {fragment} has no likelihood summary. ({p} : {pType})")
            return 0
        fragmentSummaries = {returnType: ls}  # Cache to avoid expensive recalculations.

        # Only consider tasks that have programs that can solve them.
        frontiers = [frontier for frontier in self.allFrontiers if not frontier.empty]
        if not frontiers:
            return 0

        # Save compute by tracking both numerator and denominator due to shared term.
        numerator = 0
        denominator = 0
        for frontier in frontiers:
            grammar = self.taskGrammars[frontier.task]
            if fromRoot:
                fragmentL = math.exp(ls.logLikelihood(grammar))
            else: 
                fragmentL = math.exp(self.marginalizedLL(ls, returnType, grammar))
            if frontier.empty:
                chunkGivenTaskPr = 0
            else:
                chunkGivenTaskPr = self.chunkGivenTaskProbability(
                    fragment,
                    frontier,
                    chunkWeighting,
                    fragmentSummaries
                )
            numerator += fragmentL * chunkGivenTaskPr
            denominator += fragmentL
        
        return numerator / denominator

    @staticmethod
    def consolidate(
        grammar,
        recognitionModel,
        frontiers,
        useProgramPrimArgTypeCounts,
        fromRoot,
        chunkWeighting,
        pseudoCounts,
        arity,
        CPUs=1
    ):
        """
        Determines which, if any, functions should be chunked according to the
        DreamDecompiler then returns an updated grammar (containing those primitives 
        and re-estimated parameters) and set of frontiers (rewritten and rescored in terms
        of those primitives).
        """
        originalFrontiers = frontiers
        frontiers = [frontier for frontier in frontiers if not frontier.empty]

        ddc = DreamDecompiler(
            recognitionModel,
            frontiers,
            useProgramPrimArgTypeCounts=useProgramPrimArgTypeCounts
        )

        def fragmentChunkPr(fragment):
            chunkPr = ddc.fragmentChunkProbability(fragment, fromRoot, chunkWeighting)
            return fragment, chunkPr

        # Chunk fragments with a chunking probability >= 0.5.
        fragments = proposeFragmentsFromFrontiers(frontiers, arity, CPUs)
        fragmentChunkPrs = parallelMap(CPUs, fragmentChunkPr, fragments)
        fragmentsToChunk = [(fragment, chunkPr) for fragment, chunkPr in fragmentChunkPrs if chunkPr >= 0.5]

        eprint(f"Found {len(fragmentsToChunk)} fragments worth chunking from {len(fragments)} candidates:")
        for fragment, chunkPr in fragmentsToChunk:
            eprint(f"{chunkPr}\t{fragment}")
        fragmentsToChunk = [fragment for fragment, _ in fragmentsToChunk]

        # Rewrite frontiers in terms of new fragments, starting with largest.
        if fragmentsToChunk:
            fragmentsToChunk.sort(key=fragmentSize, reverse=True)
            for fragment in fragmentsToChunk:
                frontiers = [RewriteFragments.rewriteFrontier(frontier, fragment) for frontier in frontiers]
            
            # Update grammar and frontiers.
            newPrimitives = [defragment(fragment) for fragment in fragmentsToChunk]
            newGrammar =  Grammar.uniform(
                grammar.primitives + newPrimitives,
                continuationType=grammar.continuationType
            )
            newGrammar = newGrammar.insideOutside(frontiers, pseudoCounts)
            frontiers = [newGrammar.rescoreFrontier(f) for f in frontiers]
        else:
            newGrammar = grammar

        # Add back in empty frontiers.
        frontiers = {f.task: f for f in frontiers}
        frontiers = [frontiers.get(f.task, f) for f in originalFrontiers]

        return newGrammar, frontiers

    
    ################################################################################
    ### REDOING CHUNK PROBABILITY METHODS TO USE VERSION SPACE PROPOSALS INSTEAD ###
    ################################################################################


    def chunkGivenTaskProbabilityVS(
        self,
        fragment,
        invented,
        frontier,
        chunkWeighting,
        fragmentSummaries=None
    ):
        """
        Modification of `chunkGivenTaskProbability` to work within the version space
        consolidation.

        Returns the probability of chunking a fragment on a certain task (P(c|f, Q, x))
        by marginalising over the programs in the task's frontier.

        chunkWeighting: Whether the probability of chunking a fragment for a given task
        should be weighted by the raw probabilities of the recognition model generating
        each program in the beam ("raw") and thus a lower bound to true marginalisation
        over all programs, just propotionate to the recognition model beam probabilities
        ("prop") or if a unifrom distribution over the programs in the beam should be
        used ("uniform").
        """
        assert chunkWeighting in {"raw", "prop", "uniform"}
        fragmentSummaries = fragmentSummaries if fragmentSummaries is not None else {}
        taskGrammar = self.taskGrammars[frontier.task]
        request = frontier.task.request

        concreteFragment = invented
        rewrittenFrontier = frontier

        chunkGivenTaskPr = 0
        normaliser = 0
        for e in rewrittenFrontier.entries:
            if not self.programContainsFragment(e.program, concreteFragment):
                continue  # Probability of chunking on program is 0.

            if e.program == concreteFragment:
                # If rewritten program is equal to fragment then chunking given program probability
                # is 1 and probability of program is just the probability of the fragment.
                if chunkWeighting == "uniform":
                    chunkGivenTaskPr += 1
                else:
                    ls = self.fragmentSummariesForRequests(fragment, [request.returns()], fragmentSummaries)[0]
                    programL = math.exp(ls.logLikelihood(taskGrammar))
                    chunkGivenTaskPr += programL
                    normaliser += programL
                continue

            # Calculate LL of full program and fragment uses part of the program (split for
            # dealing with the contextual or unigram case).
            if self.contextual:
                rewriteSummaryInfo = self.contextualRewriteSummaryInfo(
                    concreteFragment,
                    e.program,
                    request
                )
                programMinusFragmentLS, fragmentAsParentLSs, fragmentOccurrences = rewriteSummaryInfo
                if programMinusFragmentLS is None:
                    # Program has no likelihood summary (occurrs for programs with
                    # indices expecting abstractions) so we give 0 probability.
                    continue  
                allFragmentsLL = self.fragmentOccurrencesLL(
                    fragment,
                    fragmentOccurrences,
                    taskGrammar,
                    fragmentSummaries
                )
                # To get LL of full program we need to sum LL of fragment parts, non fragment parts
                # and the LL of generating each immediate child of the fragment in the program. Given
                # their is no parent grammar for our fragment we use the generative model.
                programMinusFragmentLL = programMinusFragmentLS.logLikelihood(taskGrammar) 
                fragmentChildrenLL = sum(ls.logLikelihood(self.generativeModel) for ls in fragmentAsParentLSs)
                programLL = allFragmentsLL + programMinusFragmentLL + fragmentChildrenLL
            else:
                programMinusFragmentLS, fragmentRequests = self.unigramRewriteSummaryInfo(
                    concreteFragment,
                    e.program,
                    request
                )
                if programMinusFragmentLS is None:
                    continue  
                fragmentLSs = self.fragmentSummariesForRequests(
                    fragment,
                    fragmentRequests,
                    fragmentSummaries
                )
                allFragmentsLL = sum(ls.logLikelihood(taskGrammar) for ls in fragmentLSs)
                programLL = allFragmentsLL + programMinusFragmentLS.logLikelihood(taskGrammar)

            if chunkWeighting == "uniform":
                chunkGivenTaskPr += (allFragmentsLL / programLL)
            else:
                programL = math.exp(programLL)
                chunkGivenTaskPr += (allFragmentsLL / programLL) *  programL
                normaliser += programL
        
        if chunkWeighting == "uniform":
            chunkGivenTaskPr /= len(rewrittenFrontier)
        elif chunkWeighting == "prop" and chunkGivenTaskPr != 0:
            chunkGivenTaskPr /= normaliser

        return chunkGivenTaskPr

    def fragmentChunkProbabilityVS(
        self,
        fragment,
        invented,
        frontiers,
        fromRoot,
        chunkWeighting
    ):
        """
        Modification of `fragmentChunkProbability` to work within the version space
        consolidation.

        Calculates the probability (belief) of how useful chunking a given fragment
        will be for the recognition model. 

        chunkWeighting: Whether the probability of chunking a fragment for a given task
        should be weighted by the raw probabilities of the recognition model generating
        each program in the beam ("raw") and thus a lower bound to true marginalisation
        over all programs, just propotionate to the recognition model beam probabilities
        ("prop") or if a unifrom distribution over the programs in the beam should be
        used ("uniform").
        """
        ls, returnType = self.fragmentLikelihoodSummary(fragment)
        if ls is None:
            p, pType = self.fragmentToProgram(fragment)
            eprint(f"Fragment {fragment} has no likelihood summary. ({p} : {pType})")
            return 0
        fragmentSummaries = {returnType: ls}  # Cache to avoid expensive recalculations.

        # Save compute by tracking both numerator and denominator due to shared term.
        numerator = 0
        denominator = 0
        for frontier in frontiers:
            taskGrammar = self.taskGrammars[frontier.task]
            if fromRoot:
                fragmentL = math.exp(ls.logLikelihood(taskGrammar))
            else: 
                fragmentL = math.exp(self.marginalizedLL(ls, returnType, taskGrammar))
            if frontier.empty:
                chunkGivenTaskPr = 0
            else:
                chunkGivenTaskPr = self.chunkGivenTaskProbabilityVS(
                    fragment,
                    invented,
                    frontier,
                    chunkWeighting,
                    fragmentSummaries
                )
            numerator += fragmentL * chunkGivenTaskPr
            denominator += fragmentL
        
        return numerator / denominator

    def filterCandidateChunkPrs(self, candidateChunkPrs):
        """
        Return a filtered list of (candidate, chunkPr, fragment) triplets containint only
        one of any slight variants. Done by looking at those candidates that have the
        same nodes up to argument numbering and thus same chunkPr. For example, only the
        first seen of the two functions f(x, y) = x + y and g(x, y) = y + x would be kept.
        """

        def primitiveCounts(fragment):
            counts = {p: 0 for p in self.generativeModel.primitives}
            for _, p in fragment.walk():
                if p in counts:
                    counts[p] += 1
            return counts

        # We start with narrowing problem to those with same chunk probability because if
        # they are slight variants in the sense described they will have same chunk probability.
        # Then we confirm they have the same nodes (other than variable indices).
        filteredCandidateChunkPrs = []
        seenChunkPrs = {}
        for candidate, chunkPr, fragment in candidateChunkPrs:
            if fragment is None:
                continue
            p = round(chunkPr, 12)
            if p not in seenChunkPrs:
                seenChunkPrs[p] = [primitiveCounts(fragment)]
                filteredCandidateChunkPrs.append((candidate, chunkPr, fragment))
            else:
                counts = primitiveCounts(fragment)
                if counts not in seenChunkPrs[p]:
                    seenChunkPrs[p].append(counts)
                    filteredCandidateChunkPrs.append((candidate, chunkPr, fragment))
        return filteredCandidateChunkPrs

    @staticmethod
    def consolidateVS(
        grammar,
        recognitionModel,
        frontiers,
        useProgramPrimArgTypeCounts,
        fromRoot,
        chunkWeighting,
        numConsolidate,
        maximumFrontier,
        pseudoCounts=1.,
        arity=3,
        topK=2,
        topI=300,
        bs=1000000,
        CPUs=1
    ):
        """
        Determines which, if any, functions should be chunked according to the
        DreamDecompiler with using version spaces to find candidate functions
        then returns an updated grammar (containing those primitives and re-estimated
        parameters) and set of frontiers (rewritten and rescored in terms of those
        primitives).
        """

        if type(numConsolidate) == int and numConsolidate == 0:
            # Nothing to do if request is to consolidate 0.
            originalFrontiers = frontiers
            frontiers = [frontier for frontier in frontiers if not frontier.empty]
            if not frontiers:
                return grammar, originalFrontiers
            grammar = grammar.insideOutside(frontiers, pseudoCounts)
            frontiers = [grammar.rescoreFrontier(f) for f in frontiers]
            frontiers = {f.task: f for f in frontiers}
            frontiers = [frontiers.get(f.task, f) for f in originalFrontiers]
            return grammar, frontiers

        # The following code starts off similar (some parts copy pasted exactly) to
        # `induceGrammar_Beta` in vs.py, but then uses RecogntionCompiler for choosing
        # from candidates instead and merging resulting frontiers.
        originalFrontiers = frontiers
        frontiers = [frontier for frontier in frontiers if not frontier.empty]

        def restrictFrontiers():
            return parallelMap(
                1,
                lambda f: grammar.rescoreFrontier(f).topK(topK),
                frontiers,
                memorySensitive=True,
                chunksize=1,
                maxtasksperchild=1
            )

        restrictedFrontiers = restrictFrontiers()

        v = VersionTable(typed=False, identity=False)
        with timing("constructed %d-step version spaces"%arity):
            versions = [[v.superVersionSpace(v.incorporate(e.program), arity) for e in f]
                        for f in restrictedFrontiers ]
            eprint("Enumerated %d distinct version spaces"%len(v.expressions))
        
        candidates = v.bestInventions(versions, bs=bs)[:topI]
        eprint("Only considering the top %d candidates"%len(candidates))

        # Clean caches that are no longer needed
        v.recursiveTable = [None]*len(v)
        v.inhabitantTable = [None]*len(v)
        v.functionInhabitantTable = [None]*len(v)
        v.substitutionTable = {}
        gc.collect()

        # At this point we have found candidates for our restricted frontiers, and
        # want to get chunk probabilities for each candidate.
        ddc = DreamDecompiler(
            recognitionModel,
            restrictedFrontiers,
            useProgramPrimArgTypeCounts=useProgramPrimArgTypeCounts,
        )

        if chunkWeighting is None:
            eprint("Scoring candidates based on their expecteced likelihood over tasks.")
        else:
            eprint(f"Scoring candidates using their '{chunkWeighting}' chunk probability.")

        def candidateChunkPr(candidate):
            try:
                if chunkWeighting is None:
                    # Find expected likelihood of recognition model generating fragment. 
                    fragment = next(v.extract(candidate))
                    chunkPr = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=fromRoot))
                    return candidate, chunkPr, fragment

                invented, rewrittenFrontiers = v.rewriteFrontiersWithInvention(
                    candidate,
                    restrictedFrontiers,
                )
            except Exception:
                # As in DreamCoder: occurs if candidate is not well typed which is
                # expected and more effecient to filter out than avoid proposing them.
                return candidate, 0, None
        
            fragment = next(v.extract(candidate))
            chunkPr = ddc.fragmentChunkProbabilityVS(
                fragment,
                invented,
                rewrittenFrontiers,
                fromRoot=fromRoot,
                chunkWeighting=chunkWeighting,
            )
            return candidate, chunkPr, fragment

        with timing("Found chunk probability of all candidates"):
            candidateChunkPrs = parallelMap(
                8,
                candidateChunkPr,
                candidates,
                memorySensitive=True,
                chunksize=1,
                maxtasksperchild=1
            )

        candidateChunkPrs = ddc.filterCandidateChunkPrs(candidateChunkPrs)

        eprint(f"Candidate chunk probabilities ({chunkWeighting}):")
        chunkPrs = [chunkPr for _, chunkPr, _ in candidateChunkPrs]
        chunkPrs.sort(reverse=True)
        eprint(chunkPrs)

        # Find candidates worth chunking.
        if type(numConsolidate) == int:
            # Chunk top given candidates.
            eprint(f"Chunking top {numConsolidate} candidates:")
            candidateChunkPrs.sort(key=lambda c: c[1], reverse=True)
            candidatesToChunk = []
            for candidate, chunkPr, fragment in candidateChunkPrs[:numConsolidate]:
                candidatesToChunk.append(candidate)
                eprint(f"{chunkPr}\t{fragment}")
        elif type(numConsolidate) == float:
            # Chunk all candidates with probability above given threshold.
            eprint(f"Chunking candidates with a chunk probability >= {numConsolidate}:")
            candidatesToChunk = []
            for candidate, chunkPr, fragment in candidateChunkPrs:
                if chunkPr >= numConsolidate:
                    candidatesToChunk.append(candidate)
                    eprint(f"{chunkPr}\t{fragment}")
        else:
            raise TypeError

        if not candidatesToChunk:
            eprint("The top 3 candidates were:")
            top3Candidates = sorted(candidateChunkPrs, key=lambda c: c[1], reverse=True)[:3]
            for _, chunkPr, fragment in top3Candidates:
                eprint(f"{chunkPr}\t{fragment}")
            return grammar, originalFrontiers

        # We want to rewrite all frontiers in terms of the new candidates so must
        # first construct version space for all.
        with timing("Constructed version space for entire frontiers"):
            for f in frontiers:
                for e in f:
                    v.superVersionSpace(v.incorporate(e.program), arity)

        if len(candidatesToChunk) == 1:
            grammar, frontiers = v.addInventionToGrammar(
                candidatesToChunk[0],
                grammar,
                frontiers,
                pseudoCounts=pseudoCounts
            )
        else:
            # We can't use the same version table to repeatedly rewrite all frontiers for each
            # candidate (and can't construct new one as candidate indices would change). Therefore,
            # we instead rewrite the frontiers in terms of each candidate individually then merge:
            # for each program of each frontier, taking a program that was able to be refactored to
            # use new invention (where higher chunkPr gets precedence) and if not using original.
            allCandidateFrontiers = parallelMap(
                min(4, len(candidatesToChunk)),
                lambda c: v.addInventionToGrammar(c, grammar, frontiers, pseudoCounts=pseudoCounts),
                candidatesToChunk,
                memorySensitive=True,
                chunksize=1,
                maxtasksperchild=1
            )
            # Switch from list of all task frontiers for each candidate to
            # list of candidate frontiers (and original) for each task.
            inventions = []
            allPossibleFrontiers = [[None] * len(candidatesToChunk) + [f] for f in frontiers]
            for candidateIndex, (candidateGrammar, candidateFrontiers) in enumerate(allCandidateFrontiers):
                inventions.append(candidateGrammar.primitives[0])
                for frontierIndex, f in enumerate(candidateFrontiers): 
                    # Ensure entries are sorted before storing.
                    allPossibleFrontiers[frontierIndex][candidateIndex] = f.topK(maximumFrontier)

            # Merge frontiers from each candidate.
            mergedFrontiers = [Frontier([], f.task) for f in frontiers]
            for i in range(len(frontiers)):
                usedPrograms = set()
                # Interleave the entries of possible frontiers: so for frontiers with entries
                # ABC and DEF we loop through the entries in order ADBECF.
                for e in itertools.chain(*zip(*(f.entries for f in allPossibleFrontiers[i]))):
                    if e.program not in usedPrograms:
                        usedPrograms.add(e.program)
                        mergedFrontiers[i].entries.append(e)
                        if len(mergedFrontiers[i]) == maximumFrontier:
                            break
            
            # Calculate parameters of grammar with all new inventions and score merged frontiers
            grammar = Grammar.uniform(
                grammar.primitives + inventions,
                continuationType=grammar.continuationType
            )
            grammar = grammar.insideOutside(mergedFrontiers, pseudoCounts)
            frontiers = [grammar.rescoreFrontier(f) for f in mergedFrontiers]

        # Add back in empty frontiers.
        frontiers = {f.task: f for f in frontiers}
        frontiers = [frontiers.get(f.task, f) for f in originalFrontiers]

        return grammar, frontiers


# Convenient container for storing evaluation results.
FragmentRCResult = namedtuple(
    "FragmentRCResult",
    [
        "fragment",
        "expectedL",
        "expectedLRoot",
        "chunkPr",
        "chunkPrRoot",
        "chunkPrU",
        "chunkPrURoot",
        "chunkPrProp",
        "chunkPrPropRoot",
    ]
)

# Exploratory functions for comparing differences in the approaches.

def evaluateViewResults(
    result,
    frontiers,
    arity,
    useProgramPrimArgTypeCounts=True,
    CPUs=1,
    topKStored=10
):
    """
    Evaluate and store various aspects of the recognition compiler on the result
    of a DreamCoder iteration to compare differences in abstraction preferences.
    The results of various variations are all calcualted at once as the recognition
    compiler has no influence on the system for view results.
    """
    ddc = DreamDecompiler(
        result.recognitionModel,
        frontiers,
        useProgramPrimArgTypeCounts=useProgramPrimArgTypeCounts
    )

    fragments = proposeFragmentsFromFrontiers(ddc.allFrontiers, arity, CPUs)
    eprint(f"Evaluating RC variant score and ranks for {len(fragments)} proposed fragments.")

    def scoreFragment(fragment):
        concreteFragment = defragment(fragment)
        expectedL = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=False))
        expectedLRoot = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=True))
        chunkPr = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="raw")
        chunkPrRoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="raw")
        chunkPrU = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="uniform")
        chunkPrURoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="uniform")
        chunkPrProp = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="prop")
        chunkPrPropRoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="prop")
        return (
            concreteFragment,
            expectedL,
            expectedLRoot,
            chunkPr,
            chunkPrRoot,
            chunkPrU,
            chunkPrURoot,
            chunkPrProp,
            chunkPrPropRoot
        )
    
    fragmentScores = parallelMap(CPUs, scoreFragment, fragments) 
    expectedLs = sorted(fragmentScores, reverse=True, key=itemgetter(1))
    expectedLsRoot = sorted(fragmentScores, reverse=True, key=itemgetter(2))
    chunkPrs = sorted(fragmentScores, reverse=True, key=itemgetter(3))
    chunkPrsRoot = sorted(fragmentScores, reverse=True, key=itemgetter(4))
    chunkPrUs = sorted(fragmentScores, reverse=True, key=itemgetter(5))
    chunkPrUsRoot = sorted(fragmentScores, reverse=True, key=itemgetter(6))
    chunkPrProps = sorted(fragmentScores, reverse=True, key=itemgetter(7))
    chunkPrPropsRoot = sorted(fragmentScores, reverse=True, key=itemgetter(8))

    def createRCResult(fragment):

        def getScoreAndRank(target, sortedScores, scoreIndex):
            for i, fragmentScore in enumerate(sortedScores):
                if target == fragmentScore[0]:
                    return fragmentScore[scoreIndex], i

        concreteFragment = defragment(fragment)
        rcResult = FragmentRCResult(
            concreteFragment,
            getScoreAndRank(concreteFragment, expectedLs, 1),
            getScoreAndRank(concreteFragment, expectedLsRoot, 2),
            getScoreAndRank(concreteFragment, chunkPrs, 3),
            getScoreAndRank(concreteFragment, chunkPrsRoot, 4),
            getScoreAndRank(concreteFragment, chunkPrUs, 5),
            getScoreAndRank(concreteFragment, chunkPrUsRoot, 6),
            getScoreAndRank(concreteFragment, chunkPrProps, 7),
            getScoreAndRank(concreteFragment, chunkPrPropsRoot, 8),
        )
        return rcResult

    # For each proposed fragment, find its score and rank in each list: only
    # keeping the fragments that are in the top k of any scoring.
    proposalScores = parallelMap(CPUs, createRCResult, fragments)
    proposalScores = [s for s in proposalScores if any(rank < topKStored for _, rank in s[1:])]

    def rankScore(targetScore, sortedScores, scoreIndex):
        if targetScore == 0:
            return -1
        for i, fragmentScore in enumerate(sortedScores):
            if targetScore >= fragmentScore[scoreIndex]:
                return i

    # For each fragment chunked by DreamCoder, fint its score and rank in the
    # proposal lists (not taking other DreamCoder positions into account).
    prevGrammar = result.grammars[-2]
    newGrammar = result.grammars[-1]
    dreamCoderPrimitives = list(set(newGrammar.primitives) - set(prevGrammar.primitives))
    dreamCoderScores = []
    for primitive in dreamCoderPrimitives:
        fragment = primitive.body  # Remove invention
        expectedL = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=False))
        expectedLRoot = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=True))
        chunkPr = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="raw")
        chunkPrRoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="raw")
        chunkPrU = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="uniform")
        chunkPrURoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="uniform")
        chunkPrProp = ddc.fragmentChunkProbability(fragment, fromRoot=False, chunkWeighting="prop")
        chunkPrPropRoot = ddc.fragmentChunkProbability(fragment, fromRoot=True, chunkWeighting="prop")
        rcResult = FragmentRCResult(
            primitive,
            (expectedL, rankScore(expectedL, expectedLs, 1)),
            (expectedLRoot, rankScore(expectedLRoot, expectedLsRoot, 2)),
            (chunkPr, rankScore(chunkPr, chunkPrs, 3)),
            (chunkPrRoot, rankScore(chunkPrRoot, chunkPrsRoot, 4)),
            (chunkPrU, rankScore(chunkPrU, chunkPrUs, 5)),
            (chunkPrURoot, rankScore(chunkPrURoot, chunkPrUsRoot, 6)),
            (chunkPrProp, rankScore(chunkPrProp, chunkPrProps, 7)),
            (chunkPrPropRoot, rankScore(chunkPrPropRoot, chunkPrPropsRoot, 8))
        )
        dreamCoderScores.append(rcResult)

    # Store results on given ECResult.
    rcViewResults = {
        "useProgramPrimArgTypeCounts": ddc.useProgramPrimArgTypeCounts,
        "numProposedFragments": len(fragments),
        "proposalScores": proposalScores,
        "dreamCoderScores": dreamCoderScores,
    } 
    result.rcViewResults.append(rcViewResults)

    # Print summary of results.
    eprint("Of %d proposed fragments, %d have a likelihood score." % 
            (len(fragments), len([s for s in fragmentScores if s[1] != 0])))
    eprint("Of %d proposed fragments, %d have a chunking score." % 
            (len(fragments), len([s for s in fragmentScores if s[3] != 0])))
    eprint("useProgramPrimArgTypeCounts:", ddc.useProgramPrimArgTypeCounts)
    eprint("Showing top 3 fragments for each DreamDecompiler score:")
    def printTopFragments(scoreName, sortedScores, scoreIndex):
        eprint(scoreName + ":")
        for fragmentScore in sortedScores[:3]:
            eprint("%.07f\t%s" % (fragmentScore[scoreIndex], fragmentScore[0]))
        eprint()
    printTopFragments("Expected likelihood over tasks", expectedLs, 1)
    printTopFragments("Expected likelihood over tasks (root)", expectedLsRoot, 2)
    printTopFragments("Probability of chunking (LB)", chunkPrs, 3)
    printTopFragments("Probability of chunking (LB-root)", chunkPrsRoot, 4)
    printTopFragments("Probability of chunking (U)", chunkPrUs, 5)
    printTopFragments("Probability of chunking (U-root)", chunkPrUsRoot, 6)
    printTopFragments("Probability of chunking (Prop)", chunkPrProps, 7)
    printTopFragments("Probability of chunking (Prop-root)", chunkPrPropsRoot, 8)
    eprint("Number of fragments with chunk probability >= 0.5:", 
            len([s for s in fragmentScores if s[7] >= 0.5]), "\n")

    eprint("DreamCoder's primitive DreamDecompiler score and ranks:")
    def printScoreAndRank(scoreName, primitiveScore, scoreIndex):
        score, rank = primitiveScore[scoreIndex]
        eprint("%s\t%.07f\t%d" % (scoreName.ljust(40), score, rank))
    for primitiveScore in dreamCoderScores:
        eprint(primitiveScore[0])
        printScoreAndRank("Expected likelihood", primitiveScore, 1)
        printScoreAndRank("Expected likelihood (root)", primitiveScore, 2)
        printScoreAndRank("Probability of chunking (LB)", primitiveScore, 3)
        printScoreAndRank("Probability of chunking (LB-root)", primitiveScore, 4)
        printScoreAndRank("Probability of chunking (U)", primitiveScore, 5)
        printScoreAndRank("Probability of chunking (U-root)", primitiveScore, 6)
        printScoreAndRank("Probability of chunking (Prop)", primitiveScore, 7)
        printScoreAndRank("Probability of chunking (Prop-root)", primitiveScore, 8)
        eprint()

    return rcViewResults


def evaluateViewResultsVS(
    result,
    frontiers,
    arity,
    useProgramPrimArgTypeCounts,
    topK=2,
    topI=300,
    bs=1000000,
    CPUs=1,
    topKStored=10
):
    """
    Evaluate and store various aspects of the recognition compiler on the result
    of a DreamCoder iteration to compare differences in abstraction preferences.
    The results of various variations are all calcualted at once as the recognition
    compiler has no influence on the system for view results. Proposals are made
    using version spaces.
    """

    grammar = result.grammars[-2] 
    frontiers = [frontier for frontier in frontiers if not frontier.empty]

    def restrictFrontiers():
        return parallelMap(
            1,
            lambda f: grammar.rescoreFrontier(f).topK(topK),
            frontiers,
            memorySensitive=True,
            chunksize=1,
            maxtasksperchild=1
        )

    restrictedFrontiers = restrictFrontiers()

    v = VersionTable(typed=False, identity=False)
    with timing("constructed %d-step version spaces"%arity):
        versions = [[v.superVersionSpace(v.incorporate(e.program), arity) for e in f]
                    for f in restrictedFrontiers ]
        eprint("Enumerated %d distinct version spaces"%len(v.expressions))
    
    candidates = v.bestInventions(versions, bs=bs)[:topI]
    eprint("Only considering the top %d candidates"%len(candidates))

    # Clean caches that are no longer needed
    v.recursiveTable = [None]*len(v)
    v.inhabitantTable = [None]*len(v)
    v.functionInhabitantTable = [None]*len(v)
    v.substitutionTable = {}
    gc.collect()

    # At this point we have found candidates for our restricted frontiers, and
    # want to get chunk probabilities for each candidate.
    ddc = DreamDecompiler(
        result.recognitionModel,
        restrictedFrontiers,
        useProgramPrimArgTypeCounts=useProgramPrimArgTypeCounts,
    )

    def inventedTaskUses(invented, frontiers):
        uses = 0
        for f in frontiers:
            for e in f.entries:
                if ddc.programContainsFragment(e.program, invented):
                    uses += 1
                    break
        return uses

    def scoreCandidate(candidate):
        try:
            invented, rewrittenFrontiers = v.rewriteFrontiersWithInvention(
                candidate,
                restrictedFrontiers,
            )
        except InferenceFailure:
            # As in DreamCoder: occurs if candidate is not well typed which is
            # expected and more effecient to filter out than avoid proposing them.
            return None

        # Only scoring for probabilities not from root.
        fragment = next(v.extract(candidate))
        expectedL = math.exp(ddc.expectedLLOverTasks(fragment, fromRoot=False))
        chunkPr = ddc.fragmentChunkProbabilityVS(
            fragment,
            invented,
            rewrittenFrontiers,
            fromRoot=False,
            chunkWeighting="raw",
        )
        chunkPrU = ddc.fragmentChunkProbabilityVS(
            fragment,
            invented,
            rewrittenFrontiers,
            fromRoot=False,
            chunkWeighting="uniform",
        )
        chunkPrProp = ddc.fragmentChunkProbabilityVS(
            fragment,
            invented,
            rewrittenFrontiers,
            fromRoot=False,
            chunkWeighting="prop",
        )

        # Store number of tasks that use the invention.
        uses = inventedTaskUses(invented, rewrittenFrontiers)

        return (invented, expectedL, chunkPr, chunkPrU, chunkPrProp, uses)

    with timing("Scored all candidates"):
        candidateScores = parallelMap(
            CPUs,
            scoreCandidate,
            candidates,
            memorySensitive=True,
            chunksize=1,
            maxtasksperchild=1
        )
        candidateScores = [s for s in candidateScores if s is not None]
    
    expectedLs = sorted(candidateScores, reverse=True, key=itemgetter(1))
    chunkPrs = sorted(candidateScores, reverse=True, key=itemgetter(2))
    chunkPrUs = sorted(candidateScores, reverse=True, key=itemgetter(3))
    chunkPrProps = sorted(candidateScores, reverse=True, key=itemgetter(4))

    def getScoreAndRank(target, sortedScores, scoreIndex):
        for i, s in enumerate(sortedScores):
            if target == s[0]:
                return s[scoreIndex], i

    candidateResults = []
    for s in candidateScores:
        invented = s[0]
        rcResult = FragmentRCResult(
            invented,
            getScoreAndRank(invented, expectedLs, 1),
            (None, len(candidateScores)),
            getScoreAndRank(invented, chunkPrs, 2),
            (None, len(candidateScores)),
            getScoreAndRank(invented, chunkPrUs, 3),
            (None, len(candidateScores)),
            getScoreAndRank(invented, chunkPrProps, 4),
            (None, len(candidateScores)),
        )
        candidateResults.append(rcResult)
    # Store top candidates of each list only.
    candidateResults = [s for s in candidateResults if any(rank < topKStored for _, rank in s[1:])]

    # For each fragment chunked by DreamCoder, fint its score and rank in the
    # proposal lists (not taking other DreamCoder positions into account).
    newGrammar = result.grammars[-1]
    dreamCoderPrimitives = list(set(newGrammar.primitives) - set(grammar.primitives))
    dreamCoderResults = []
    for primitive in dreamCoderPrimitives:
        if not any(primitive == s[0] for s in candidateScores):
            # DreamCoder primitive is not a candidate that we considered (can
            # happen as they make recursive induction calls in a single iteration,
            # but we can't as we'd need to retriain the recognition model).
            dreamCoderResults.append(FragmentRCResult(primitive,  *((None,) * 8)))
        else:
            rcResult = FragmentRCResult(
                primitive,
                getScoreAndRank(primitive, expectedLs, 1),
                (None, len(candidateScores)),
                getScoreAndRank(primitive, chunkPrs, 2),
                (None, len(candidateScores)),
                getScoreAndRank(primitive, chunkPrUs, 3),
                (None, len(candidateScores)),
                getScoreAndRank(primitive, chunkPrProps, 4),
                (None, len(candidateScores)),
            )
            dreamCoderResults.append(rcResult)

    # Store results on given ECResult.
    rcViewResults = {
        "useProgramPrimArgTypeCounts": ddc.useProgramPrimArgTypeCounts,
        "candidateResults": candidateResults,
        "dreamCoderResults": dreamCoderResults,
    } 
    result.rcViewResults.append(rcViewResults)

    def reduceScores(sortedScores, scoreIndex):
        reducedScores = []
        scores = set()
        for fragmentScore in sortedScores:
            if fragmentScore[scoreIndex] not in scores:
                scores.add(fragmentScore[scoreIndex])
                reducedScores.append(fragmentScore)
        return reducedScores

    eprint("Showing top 10 fragments for each DreamDecompiler score:")
    def printTopFragments(scoreName, sortedScores, scoreIndex, reducedScores):
        eprint(scoreName + ":")
        for fragmentScore in sortedScores[:10]:
            eprint("%.07f (Uses: %d)\t%s" % (fragmentScore[scoreIndex], fragmentScore[5], fragmentScore[0]))
        
        eprint(f"\nTop 10 of {len(reducedScores)} candidates in reduced set:")
        for fragmentScore in reducedScores[:10]:
            eprint("%.07f (Uses: %d)\t%s" % (fragmentScore[scoreIndex], fragmentScore[5], fragmentScore[0]))
        eprint()
    printTopFragments("Expected likelihood over tasks", expectedLs, 1, reduceScores(expectedLs, 1))
    reducedChunkPrs = reduceScores(chunkPrs, 2)
    printTopFragments("Probability of chunking (raw)", chunkPrs, 2, reducedChunkPrs)
    printTopFragments("Probability of chunking (unifrom)", chunkPrUs, 3, reduceScores(chunkPrUs, 3))
    reducedChunkPrProps = reduceScores(chunkPrProps, 4)
    printTopFragments("Probability of chunking (prop)", chunkPrProps, 4, reducedChunkPrProps)
    eprint("Number of fragments with chunk probability (prop) >= 0.5:", 
            len([r for r in candidateScores if r[4] >= 0.5]), 
            f"({len([r for r in reducedChunkPrProps if r[4] >= 0.5])})\n")
    eprint("Number of fragments with raw probability >= 0.003:", 
            len([r for r in candidateScores if r[2] >= 0.003]),
            f"({len([r for r in reducedChunkPrs if r[2] >= 0.003])})\n")

    eprint("DreamCoder's primitive DreamDecompiler score and ranks:")
    def printScoreAndRank(scoreName, primitiveResult, scoreIndex):
        if primitiveResult[scoreIndex] is None:
            score, rank = -1, -1
        else:
            score, rank = primitiveResult[scoreIndex]
        eprint("%s\t%.07f\t%d" % (scoreName.ljust(40), score, rank))
    for r in dreamCoderResults:
        eprint(r[0])
        printScoreAndRank("Expected likelihood", r, 1)
        printScoreAndRank("Probability of chunking (LB)", r, 3)
        printScoreAndRank("Probability of chunking (U)", r, 5)
        printScoreAndRank("Probability of chunking (Prop)", r, 7)
        eprint()

    return rcViewResults
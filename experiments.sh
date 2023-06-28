# See README for setting up codebase to run experiments.

# Symbolic Regression
# DreamCoder
python bin/rational.py  -t 120  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20

# DreamDecompiler-Avg
python bin/rational.py  -t 120  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20 --compressor ddc_vs --numConsolidate 0,1,1,1,1,1,1,1,1,1

# DreamDecompiler-PC
python bin/rational.py  -t 120  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20 --compressor ddc_vs --numConsolidate 0,1,1,1,1,1,1,1,1,1 --chunkWeighting raw

# LOGO graphics
# DreamCoder
python bin/logo.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual  --taskReranker randomShuffle --taskBatchSize 20 -RS 5000 --seed $SEED --CPUs 40

# DreamDecompiler-Avg
python bin/logo.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual  --taskReranker randomShuffle --taskBatchSize 20 -RS 5000 --seed $SEED --CPUs 40 --compressor ddc_vs --numConsolidate 0,1,1,1,1,1,1,1,1,1

# DreamDecompiler-PC
python bin/logo.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual  --taskReranker randomShuffle --taskBatchSize 20 -RS 5000 --seed $SEED --CPUs 40 --compressor ddc_vs --numConsolidate 0,1,1,1,1,1,1,1,1,1 --chunkWeighting raw

# List processing
# DreamCoder
python bin/list.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20

# DreamDecompiler-Avg
python bin/list.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20 --compressor ddc_vs
# Ran with the following to match DreamCoder:
# --numConsolidate 2,3,2,0,2,0,1,0,0,1
# --numConsolidate 1,2,3,2,2,2,3,2,4,0
# --numConsolidate 1,3,2,1,0,1,1,4,1,0

# DreamDecompiler-PC
python bin/list.py --split 0.5 -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --seed $SEED --CPUs 20 --compressor ddc_vs --chunkWeighting raw
# Ran with the following to match DreamCoder:
# --numConsolidate 2,3,2,0,2,0,1,0,0,1
# --numConsolidate 1,2,3,2,2,2,3,2,4,0
# --numConsolidate 1,3,2,1,0,1,1,4,1,0

# Text editing
# DreamCoder
python bin/text.py  -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --latest --noUnfold --seed $SEED --CPUs 20

# DreamDecompiler-Avg
python bin/text.py  -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --latest --noUnfold --seed $SEED --CPUs 20 --compressor ddc_vs
# Ran with the following to match DreamCoder:
# --numConsolidate 1,1,1,1,3,2,2,2,0,3
# --numConsolidate 1,0,0,3,6,0,0,2,0,3
# --numConsolidate 0,3,2,2,2,1,0,0,2,1

# DreamDecompiler-PC
python bin/text.py  -t 720  --pseudoCounts 30 --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 --maximumFrontier 5 -i 10 -R 3600 --storeTaskMetrics --testingTimeout 600 --biasOptimal --contextual --taskReranker randomShuffle --taskBatchSize 10  --auxiliary --ensembleSize 1 -RS 5000 --latest --noUnfold --seed $SEED --CPUs 20 --compressor ddc_vs --chunkWeighting raw
# Ran with the following to match DreamCoder:
# --numConsolidate 1,1,1,1,3,2,2,2,0,3
# --numConsolidate 1,0,0,3,6,0,0,2,0,3
# --numConsolidate 0,3,2,2,2,1,0,0,2,1

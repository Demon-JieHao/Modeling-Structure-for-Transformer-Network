#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH
mkdir -p out

python thumt/bin/trainer.py  \
    --model transformer_onlstm_decoder \
    --output \
        ./out \
    --input \
        TrainData/corpus.bpe32k.en \
        TrainData/corpus.bpe32k.de \
    --vocabulary \
        TrainData/vocab.en.txt \
        TrainData/vocab.de.txt \
    --validation \
	TrainData/newstest2013.bpe.en \
    --reference \
	TrainData/newstest2013.bpe.de \
    --parameters \
        "learning_rate=1,batch_size=8192,device_list=[0,1,2,3],train_steps=200000,hidden_size=512,filter_size=2048,num_heads=8,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,update_cycle=1" \
    2>&1 | tee out/log

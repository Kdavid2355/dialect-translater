# dialect-translater

## Introduction

Dialect-translater is a project dedicated to preserving our disappearing native dialects by developing a standard language <-> dialect translator using Korean dialect natural language texts provided by AI-HUB (https://www.aihub.or.kr/).

The T5 model was utilized for this project, specifically fine-tuning the KE-T5 model. KE-T5 is a Korean language model developed using 280 GB of texts from newspapers, web texts, messengers, and spoken dialogues.

For this project, we fine-tuned KE-T5 using 1 million pairs of standard Korean language and Jeju Island dialect. We developed separate models for translating from standard language to dialect and from dialect to standard language.

The hyperparameters used were max_length = 128, batch_size = 110, lr = 1e-4, and num_epoch = 5, and the models were trained using three A6000 GPUs.

To evaluate performance, we used the BLEU Score, achieving 79.529% for 1-gram and 61.755% for 4-gram.

The results of the project can be viewed at https://kdavid2355.github.io/dialect-translater/




## Related Articles

T5 Paper: https://arxiv.org/pdf/1910.10683.pdf

(Further details will be summarized in a blog post soon)

# KERIS_TRASH
KERIS trash recognition challenge

Dataset: `recycle_challenge`
* num_test_images: 3000
* num_test_classes: 8

Pushing dataset (at the root of the folder you want to upload):

How to run:

```bash
nsml run -d recycle_challenge -g 1 --memory 12G --shm-size 2G --cpus 8 -e main.py -a \
         "--mode train --num_epochs 5 --step_size 2"
```

How to list checkpoints saved:

```bash
nsml model ls keris_admin/recycle_challenge/1
```

How to submit:

```bash
nsml submit keris_admin/recycle_challenge/1 5
```

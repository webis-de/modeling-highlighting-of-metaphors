# Modeling-highlighting-of-metaphors

Code for the paper: Modeling Highlighting of Metaphors in Multitask Contrastive Learning Paradigms

Abstract: Metaphorical language, such as ``spending time together'', projects meaning from a source domain (here, money) to a target domain (time). Thereby, it highlights certain aspects of the target domain, such as the 'effort' behind the time investment. Highlighting aspects with metaphors (while hiding others) bridges the two domains and is the core of metaphorical meaning construction. For metaphor interpretation, linguistic theories stress that identifying the highlighted aspects is important for a better understanding of metaphors. However, metaphor research in NLP has not yet dealt with the phenomenon of highlighting. In this paper, we introduce the task of identifying the main aspect highlighted in a metaphorical sentence. Given the inherent interaction of source domains and highlighted aspects, we propose two multitask approaches - a joint learning approach and a continual learning approach - based on a finetuned contrastive learning model to jointly predict highlighted aspects and source domains. We further investigate whether (predicted) information about a source domain leads to better performance in predicting the highlighted aspects, and vice versa. Our experiments on an existing corpus suggest that, with the corresponding information, the performance to predict the other improves in terms of model accuracy in predicting highlighted aspects and source domains notably compared to the single-task baselines.

Instructions to run: 

pip install setup.py

cd project_metaphor/models

* To run all single task experiments *
python3 run_single_task_all.py

* To run all multi task experiments *
python3 run_multi_task_all.py

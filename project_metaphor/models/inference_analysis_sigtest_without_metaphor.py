from base64 import encode
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from HighlightedAspects import highlighted_classes, source_cm
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os

# from project_metaphor import PROJECT_ROOT

import wandb

RESULTS_PATH = os.path.join('../analysis/')

logging.basicConfig(level=logging.INFO)


class InferenceRankingEvaluator(SentenceEvaluator):
    def __init__(
        self,
        model,
        anchors_with_ground_truth_candidates,
        model_abbreviation,
        task='hghl',
        batch_size: int = 32
    ):

        self.model = model
        self.model_abbreviation = model_abbreviation
        self.anchors_with_ground_truth_candidates = anchors_with_ground_truth_candidates
        self.highlighted_classes = highlighted_classes
        self.source_cm = source_cm
        self.batch_size = batch_size
        self.task = task
        print('TASK IN INF EVAL', self.task, type(self.task))

    def encode(self, model, anchors, candidates):
        self.model = model
        this_task = self.task
        embeddings_anchors = self.model.encode(
            anchors, batch_size=self.batch_size,
            convert_to_numpy=True
        )
        embeddings_candidates = self.model.encode(
            candidates,
            convert_to_numpy=True
        )

        if this_task == 'hghl':
            print('In the INF EVAL HGHL')
            embeddings_highlighted = self.model.encode(
                self.highlighted_classes,
                convert_to_numpy=True
            )

        else:
            print('In the INF EVAL SCM')
            embeddings_highlighted = self.model.encode(
                self.source_cm,
                convert_to_numpy=True
            )

        assert len(embeddings_anchors) == len(embeddings_candidates)

        assert len(anchors) == len(embeddings_anchors)
        self.ANCHORS_WITH_EMBEDDINGS = dict(zip(anchors, embeddings_anchors))

        assert len(candidates) == len(embeddings_candidates)
        self.CANDIDATES_WITH_EMBEDDINGS = dict(
            zip(candidates, embeddings_candidates))

        if self.task == 'hghl':
            self.HIGHLIGHTED_WITH_EMBEDDINGS = dict(
                zip(self.highlighted_classes, embeddings_highlighted))
        else:
            self.HIGHLIGHTED_WITH_EMBEDDINGS = dict(
                zip(self.source_cm, embeddings_highlighted))

        print('\nLength of list of generated embeddings is equal to the length of the list of items\n')

        return (self.ANCHORS_WITH_EMBEDDINGS, self.CANDIDATES_WITH_EMBEDDINGS, self.HIGHLIGHTED_WITH_EMBEDDINGS)

    def compare(self, anchor, anchor_embedding, ground_truth_embeddings):
        result = dict()
        for candidate, candidate_embedding in ground_truth_embeddings.items():
            assert len(anchor_embedding) == len(candidate_embedding)
            distance = paired_cosine_distances(anchor_embedding.reshape(
                1, -1), candidate_embedding.reshape(1, -1))
            if anchor not in result.keys():
                result[anchor] = [(candidate, distance[0])]
            else:
                result[anchor].append((candidate, distance[0]))

        return result

    def get_most_similar_candidate(self, anchor_distance_scores):
        self.anchor_distance_scores = anchor_distance_scores
        candidate_scores = next(iter(self.anchor_distance_scores.values()))
        
        # print('\n')
        # print(anchor)
        # print('\n')
        # print(candidate_scores)
        # print('\n')
        
        most_similar_candidate = min(candidate_scores, key=lambda t: t[1])
        # print('\n')
        # print(most_similar_candidate)
        # print('\n')
        return most_similar_candidate[0]

    def get_most_similar_three_candidates(self,anchor_distance_scores):
            self.anchor_distance_scores = anchor_distance_scores
            
            candidate_scores = next(iter(self.anchor_distance_scores.values()))
            most_similar_candidate_1 = min(candidate_scores, key = lambda t: t[1])
            # print('\n SCORES 1 and len : ',candidate_scores,len(candidate_scores),'\n')
            candidate_scores.remove(most_similar_candidate_1) 
            # print('\n CAN 1 : ',most_similar_candidate_1,'\n')
            
            most_similar_candidate_2 = min(candidate_scores, key = lambda t: t[1])
            # print('\n SCORES 2 and len : ',candidate_scores,len(candidate_scores),'\n')
            candidate_scores.remove(most_similar_candidate_2)
            # print('\n CAN 2 : ',most_similar_candidate_2,'\n')

            most_similar_candidate_3 = min(candidate_scores, key = lambda t: t[1])
            # print('\n CAN 3 : ',most_similar_candidate_3,'\n')
            
            return [most_similar_candidate_1[0],most_similar_candidate_2[0],most_similar_candidate_3[0]] 

    def get_most_similar_five_candidates(self,anchor_distance_scores):
        self.anchor_distance_scores = anchor_distance_scores
        
        candidate_scores = next(iter(self.anchor_distance_scores.values()))
        most_similar_candidate_1 = min(candidate_scores, key = lambda t: t[1])
        # print('\n SCORES 1 and len : ',candidate_scores,len(candidate_scores),'\n')
        candidate_scores.remove(most_similar_candidate_1) 
        # print('\n CAN 1 : ',most_similar_candidate_1,'\n')
        
        most_similar_candidate_2 = min(candidate_scores, key = lambda t: t[1])
        # print('\n SCORES 2 and len : ',candidate_scores,len(candidate_scores),'\n')
        candidate_scores.remove(most_similar_candidate_2)
        # print('\n CAN 2 : ',most_similar_candidate_2,'\n')

        most_similar_candidate_3 = min(candidate_scores, key = lambda t: t[1])
        # print('\n CAN 3 : ',most_similar_candidate_3,'\n')

        candidate_scores.remove(most_similar_candidate_3)
        most_similar_candidate_4 = min(candidate_scores, key = lambda t: t[1])
        # print('\n CAN 3 : ',most_similar_candidate_3,'\n')
        candidate_scores.remove(most_similar_candidate_4)
        most_similar_candidate_5 = min(candidate_scores, key = lambda t: t[1])
        # print('\n CAN 3 : ',most_similar_candidate_3,'\n')
        return [most_similar_candidate_1[0],most_similar_candidate_2[0],most_similar_candidate_3[0],most_similar_candidate_4[0],most_similar_candidate_5[0]]

    def get_all_anchors_most_similar_candidate(self, anchor_with_embeddings, highlighted_with_embeddings):
        result = dict()
        for anchor, embedding in anchor_with_embeddings.items():
            anchor_distance_scores = self.compare(
                anchor, embedding, highlighted_with_embeddings)
            most_similar_candidate = self.get_most_similar_candidate(
                anchor_distance_scores)
            # most_similar_three_candidates = self.get_most_similar_three_candidates(
                # anchor_distance_scores)
            # most_similar_five_candidates = self.get_most_similar_five_candidates(
                # anchor_distance_scores)
            result[anchor] = most_similar_candidate

        return result

    def get_all_anchors_most_similar_three_candidates(self,anchor_with_embeddings,highlighted_with_embeddings):
        result = dict()
        for anchor,embedding in anchor_with_embeddings.items():
            anchor_distance_scores = self.compare(anchor, embedding,highlighted_with_embeddings)
            most_similar_candidates = self.get_most_similar_three_candidates(anchor_distance_scores)
            result[anchor] = most_similar_candidates

        # print('\ntop3func : ',most_similar_candidates,'\n')
        return result
    
    def get_all_anchors_most_similar_five_candidates(self,anchor_with_embeddings,highlighted_with_embeddings):
        result = dict()
        for anchor,embedding in anchor_with_embeddings.items():
            anchor_distance_scores = self.compare(anchor, embedding,highlighted_with_embeddings)
            most_similar_candidates = self.get_most_similar_five_candidates(anchor_distance_scores)
            result[anchor] = most_similar_candidates
        # print('\ntop5func : ',most_similar_candidates,'\n')
        return result

    def __call__(self):

        self.anchors = list()
        self.candidates = list()

        print(len(self.anchors_with_ground_truth_candidates))
        for anchor, candidate in self.anchors_with_ground_truth_candidates.items():
            self.anchors.append(anchor)
            self.candidates.append(candidate)
        assert len(self.anchors) == len(self.candidates)
        anchor_with_embeddings, candidate_with_embeddings, highlighted_with_embeddings = self.encode(
            self.model, self.anchors, self.candidates)

        '''
        print(next(iter(anchor_with_embeddings.items())))
        print('\n')

        print(next(iter(candidate_with_embeddings.items())))
        print('\n')
        
        print(next(iter(highlighted_with_embeddings.items())))
        print('\n')
        '''

        print('\nDictionary with anchors,candidates and their respective embeddings have been created\n')

        final = dict()
        results = dict()
        most_similar_candidates = self.get_all_anchors_most_similar_candidate(anchor_with_embeddings,highlighted_with_embeddings)
        most_similar_three_candidates = self.get_all_anchors_most_similar_three_candidates(anchor_with_embeddings,highlighted_with_embeddings)
        most_similar_five_candidates = self.get_all_anchors_most_similar_five_candidates(anchor_with_embeddings,highlighted_with_embeddings)

        '''
        Top1 Acc
        '''
        anchors = list()
        trues = list()
        predictions = list()

        for anchor, true in self.anchors_with_ground_truth_candidates.items():
            # print(anchor)
            # print(true)
            # print('\ntrue_top1 : ',true)
            predicted = most_similar_candidates[anchor]
            # print(', pred1 : ',predicted,'\n')

            anchors.append(anchor)
            trues.append(true)
            predictions.append(predicted)

            if true not in final.keys():
                if predicted == true:
                    final[true] = {'correct':1,'total':1}
                else:
                    final[true] = {'correct':0,'total':1}
            else:
                if predicted == true:
                    final[true]['correct']+=1
                    final[true]['total']+=1
                else:
                    final[true]['total']+=1

        for label, content in final.items():
            correct = content['correct']
            total = content['total']
            acc = (correct/total)*100
            results[label]=(correct,total,acc)

        sum_correct=0; sum_total=0
        for label, content in final.items():
            correct = content['correct']
            total = content['total']
            sum_correct+=correct
            sum_total+=total

        macro_average = sum_correct/sum_total

        import pandas as pd

        data_df = ({
            'Sentence' : anchors,
            'True' : trues,
            'Prediction' : predictions
            })
        
        df = pd.DataFrame(data_df)
        df.to_csv(RESULTS_PATH+'/confmtrx-multi-predict-'+self.task+'-nopretrain-nometaphor'+'.csv', index=False)
        
        # print(final)
        # print(correct)
        # print(total)
        # print(sum_correct)
        # print(sum_total)

        '''
        Top3 Acc
        '''
        
        final3 = dict()
        results3 = dict()
        for anchor3, true3 in self.anchors_with_ground_truth_candidates.items():
            # print('true_top3 : ',true3)
            predicted3 = most_similar_three_candidates[anchor3]
            # print(', pred3 : ',predicted3,'\n')
            if true3 not in final3.keys():
                if true3 in predicted3:
                    final3[true3] = {'correct':1,'total':1}
                else:
                    final3[true3] = {'correct':0,'total':1}
            else:
                if true3 in predicted3:
                    final3[true3]['correct']+=1
                    final3[true3]['total']+=1
                else:
                    final3[true3]['total']+=1

        for label3, content3 in final3.items():
            correct3 = content3['correct']
            total3 = content3['total']
            acc3 = (correct3/total3)*100
            results3[label3]=(correct3,total3,acc3)

        sum_correct3=0; sum_total3=0
        for label3, content3 in final3.items():
            correct3 = content3['correct']
            total3 = content3['total']
            sum_correct3+=correct3
            sum_total3+=total3

        macro_average_three = sum_correct3/sum_total3


        '''
        Top5 Acc
        '''
        
        final5 = dict()
        results5 = dict()
        for anchor5, true5 in self.anchors_with_ground_truth_candidates.items():
            # print('true_top5 : ',true5)
            predicted5 = most_similar_five_candidates[anchor5]
            # print(', pred5 : ',predicted5,'\n')
            if true5 not in final5.keys():
                if true5 in predicted5:
                    final5[true5] = {'correct':1,'total':1}
                else:
                    final5[true5] = {'correct':0,'total':1}
            else:
                if true5 in predicted5:
                    final5[true5]['correct']+=1
                    final5[true5]['total']+=1
                else:
                    final5[true5]['total']+=1

        for label5, content5 in final5.items():
            correct5 = content5['correct']
            total5 = content5['total']
            acc5 = (correct5/total5)*100
            results5[label5]=(correct5,total5,acc5)

        sum_correct5=0; sum_total5=0
        for label5, content5 in final5.items():
            correct5 = content5['correct']
            total5 = content5['total']
            sum_correct5+=correct5
            sum_total5+=total5

        macro_average_five = sum_correct5/sum_total5

        # print(final5)
        # print(correct5)
        # print(total5)
        # print(sum_correct5)
        # print(sum_total5)

        print(macro_average)
        print(macro_average_three)
        print(macro_average_five)

        try:
            os.makedirs(RESULTS_PATH, exist_ok = True)
        except OSError as error:
            print("Directory can not be created")
        
        # with open(RESULTS_PATH+'/results_'+self.task+'_multi_'+self.model_abbreviation+'.csv', 'w') as f:
        #     f.write("Label\t Correct\t Total\t Accuracy\n")
        #     for key in results.keys():
        #         f.write("%s\t %s\t %s\t %s\n" % (key, results[key][0], results[key][1], results[key][2]))
        #     f.write("------------------------------------------------------------------------\n")
        #     f.write("------------------------------------------------------------------------\n")
        #     f.write(" \t %s\t %s\t %s\n" % (sum_correct, sum_total, macro_average))

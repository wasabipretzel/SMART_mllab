#-*- coding:utf-8 -*-


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
import numpy as np

class ClassificationEval:
    def __init__(self, vqa=None, vqaRes=None, n=2):
        self.n = n 
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa 
        self.vqaRes = vqaRes
    
        if each_m == 'kt-emo':   # yjlee

            if task_type == 'single_label_classification':
                label_list = label_dict.values()      # [0, 1, 2, 3, 4, 5] 
            elif task_type == 'text2text':
                label_list = label_dict.keys()    # ['anger', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

            pred_labels = sorted(list(set(y_pred))) # predicted_y label list
            true_labels = sorted(list(set(y_true))) # true_y label list

            print("Predicted Labels : {}".format(pred_labels))
            print("True Labels : {}".format(true_labels))
            print("Label Dict : {}".format(label_list))

            not_in_label_dict_idx = []

            for i,pred in enumerate(pred_labels):
                if pred not in label_list:
                    not_in_label_dict_idx.append(i)     # not in label dict - ex. "urprise"
                    print("-> {} is not in Label Dict.".format(pred_labels[i]))
                elif pred not in true_labels:
                    not_in_label_dict_idx.append(i)     # not in true label - ex. only 5 emotions in data
                    print("-> {} is not in True Labels.".format(pred_labels[i]))


            each_f1 = [round(f * 100, 2) for f in f1_score(y_true, y_pred, average=None)]
            each_f1_in_true_label = []
            for i,x in enumerate(each_f1):
                if i in not_in_label_dict_idx:
                    pass
                else:
                    each_f1_in_true_label.append(x)
            
            result_dict['each-f1'] = (each_f1, True)
            result_dict['each-f1-true-label'] = (each_f1_in_true_label, True)
            result_dict['macro-f1-true-label'] = (np.mean(each_f1_in_true_label), True)

        elif 'class-report' in each_m:   # yjlee
            # import pdb; pdb.set_trace()

            if task_type == 'single_label_classification':
                label_name = [x for x in label_dict.keys() ]    # anger, fear, ...
                label_idx = [x for x in label_dict.values() ]  # 0,1,2 ...
                p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=label_idx)

                
            elif task_type == 'text2text':
                label_name = np.unique(y_pred)
                p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=label_name)
            a = accuracy_score(y_true, y_pred)

            p2 = [round(x * 100, 2) for x in p]
            r2 = [round(x * 100, 2) for x in r]
            f2 = [round(x * 100, 2) for x in f]
            a2 = round(a * 100, 2)

            from collections import Counter
            import json

            pm = round(np.mean([x for x in p if x!=0]) * 100, 2)
            rm = round(np.mean([x for x in r if x!=0]) * 100, 2)
            fm = round(np.mean([x for x in f if x!=0]) * 100, 2)

            if args.validation_data_path:
                eval_data_path = args.validation_data_path
            if args.test_data_path:
                eval_data_path = args.test_data_path


            with open('/home/work/nlp_ssd/user/yjlee/kt-ulm-fine-tuning/CLASS_REPORT.txt','a') as wf:
                if each_m == 'class-report-macro':
                    wf.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(args.output_path,pm,rm,fm,a2))
                else:
                    wf.write('\n### {}-{}\n'.format(args.data_set,args.output_path))
                    wf.write('# {}\n'.format( '-'.join(eval_data_path.split('-')[-2:]) ))

                    wf.write('data,precision,recall,f1,support,acc\n')
                    # print('acc,{}'.format(a2))
                    # wf.write('acc,{:.4f}\n'.format(a2))

                    for i in range(len(p)):
                        if label_name[i] in label_dict.keys():
                            print('{},{:.4f},{:.4f},{:.4f},{}'.format(label_name[i],p2[i],r2[i],f2[i],s[i]))
                            wf.write('{},{:.4f},{:.4f},{:.4f},{}\n'.format(label_name[i],p2[i],r2[i],f2[i],s[i]))
                    
                    # import pdb; pdb.set_trace()

                    print('average,{:.4f},{:.4f},{:.4f},,{:.4f}\n'.format(pm,rm,fm,a2))
                    wf.write('average,{:.4f},{:.4f},{:.4f},,{:.4f}\n'.format(pm,rm,fm,a2))

                print(classification_report(y_true, y_pred, zero_division=0))	# true가 없는 neutral의 recall=0

                if task_type =='text2text':
                    c = { k:v for k,v in Counter(y_pred).items() }
                    print(c)
                    wf.write('pred:{}\n'.format(json.dumps(c)))

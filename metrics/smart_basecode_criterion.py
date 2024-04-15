#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_dataset_info(csvfilename):
    import csv

    qa_info = {}
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            key = str(row["type"]).lower()
            if key not in qa_info.keys():
                qa_info[key] = [row["id"]]
            else:
                qa_info[key].append(row["id"])
    assert np.array([len(qa_info[key]) for key in qa_info.keys()]).sum() == 101
    return qa_info

class Criterion:
    def __init__(self):
        super(Criterion, self).__init__()
        self.monolithic = True  # just one classifier
        
        # prediction된 값을 받아서 acc도 계산하고, 
        self.puzzle_acc = {}
        #option, a, av 같이 받아와야함

    def compute_metrics(self, pred):
        """_summary_
            dataset collator에 "labels" key가 있어야 eval_loop이후 결과가 여기로 들어옴
        Args:
            pred (EvalPrediction): include pred.predictions (pred value), pred.label_ids(answer)
                pred.label_ids : [[test_samples, 10] -> tensor
                pred.predictions : [test_samples, 257] (max_val + 1) ->  tensor
        """
        #b_options : [test_samples, 5] , b_answer_labels : [test_samples]
        b_options, b_answer_labels = pred.inputs[0], pred.inputs[1]

        # print(pred.inputs)
        tot_samples_num = pred.predictions.shape[0]
        pred_max = F.softmax(torch.tensor(pred.predictions), dim=1).argmax(dim=1).cpu() #tensor [samples]
        answer_value = torch.tensor(pred.label_ids[:,0]) #tensor [samples]
        s_acc = (pred_max == answer_value).float().sum() / tot_samples_num #NOTE 전체 수로 나눠줘야함
        #approximate to option (return array bool)
        opt_approximate = self.get_option_sel_acc(pred_max, b_options, b_answer_labels, answer_value, -1) / tot_samples_num



        result = {
            "S_acc" : s_acc,
            "O_acc" : opt_approximate.sum()
        }
        return result



    def print_puzz_acc(self, puzz_acc, log=True):
        to_int = lambda x: np.array(list(x)).astype("int")
        cls_mean = lambda x, idx, pids: np.array([x[int(ii)] for ii in idx]).sum() / len(
            set(to_int(idx)).intersection(set(to_int(pids)))
        )
        num_puzzles=101
        acc_list = np.zeros(
            num_puzzles + 1,
        )
        opt_acc_list = np.zeros(
            num_puzzles + 1,
        )

        # if not os.path.exists(os.path.join(args.save_root, "results/%d/" % (gv.seed))):
        #     os.makedirs(os.path.join(args.save_root, "results/%d/" % (gv.seed)))

        if len(puzz_acc.keys()) > 10:
            for k, key in enumerate(puzz_acc.keys()):
                acc = 100.0 * puzz_acc[key][0] / puzz_acc[key][2]
                oacc = 100.0 * puzz_acc[key][1] / puzz_acc[key][2]
                acc_list[int(key)] = acc
                opt_acc_list[int(key)] = oacc
            if log:
                for t in range(1, gv.num_puzzles + 1):
                    print("%d opt_acc=%0.2f acc=%0.2f" % (t, opt_acc_list[t], acc_list[t]), end="\t")
                    if t % 5 == 0:
                        print("\n")
                print("\n\n")

                puzzles = read_dataset_info("/data/SMART101-release-v1/puzzle_type_info.csv")
                class_avg_perf = {}
                classes = ["counting", "math", "logic", "path", "algebra", "measure", "spatial", "pattern"]
                print(classes)
                for kk in classes:
                    idx_list = puzzles[kk]
                    class_avg_perf[kk] = (
                        cls_mean(acc_list, idx_list, list(puzz_acc.keys())),
                        cls_mean(opt_acc_list, idx_list, list(puzz_acc.keys())),
                    )
                    print("%0.1f/%0.1f & " % (class_avg_perf[kk][0], class_avg_perf[kk][1]), end=" ")
                print("\n\n")

            fig = plt.figure(figsize=(30, 4))
            ax = plt.gca()
            ax.bar(np.arange(1, gv.num_actual_puzz), fix_acc(acc_list[1:]))
            ax.set_xticks(np.arange(1, gv.num_actual_puzz))
            ax.set_xlabel("puzzle ids", fontsize=16)
            ax.set_ylabel("$O_{acc}$ %", fontsize=20)
            fig.tight_layout()
            plt.savefig(os.path.join(args.save_root, "results/%d/acc_perf_scores_1.png" % (gv.seed)))
            plt.close()

            fig = plt.figure(figsize=(30, 4))
            ax = plt.gca()
            ax.bar(np.arange(1, gv.num_actual_puzz), fix_acc(opt_acc_list[1:]))
            ax.set_xticks(np.arange(1, gv.num_actual_puzz))  # , [str(i) for i in np.arange(1,num_puzzles+1)])
            ax.set_xlabel("puzzle ids", fontsize=16)
            ax.set_ylabel("$S_{acc}$ %", fontsize=20)
            fig.tight_layout()
            plt.savefig(os.path.join(args.save_root, "results/%d/opt_acc_perf_scores_1.png" % (gv.seed)))
            plt.close()
        else:
            for key in puzz_acc.keys():
                acc = 100.0 * puzz_acc[key][0] / puzz_acc[key][2]
                opt_acc = 100.0 * puzz_acc[key][1] / puzz_acc[key][2]
                if log:
                    print("%s opt_acc=%0.2f acc=%0.2f" % (key, opt_acc, acc))
                acc_list[int(key)] = acc
                opt_acc_list[int(key)] = opt_acc

            plt.figure()
            plt.bar(np.arange(gv.num_puzzles + 1), acc_list)
            plt.savefig(os.path.join(args.save_root, "results/%d/acc_perf_scores.png" % (gv.seed)))
            plt.close()
            plt.figure()
            plt.bar(np.arange(gv.num_puzzles + 1), opt_acc_list)
            plt.savefig(os.path.join(args.save_root, "results/%d/opt_acc_perf_scores.png" % (gv.seed)))
            plt.close()

    def get_option_sel_acc(self, pred_ans, opts, answer, answer_values, pid):
        """converts a predicted answer to one of the given multiple choice options.
        opts is b x num_options matrix"""
        # PS setting에서는 seq puzzle이 없기 떄문에 pid값을 -1 로 넣음
        signs = np.array(["+", "-", "x", "/"])  # puzzle 58
        SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]

        def get_op_str(ii):
            return signs[int(str(ii)[0]) - 1] + str(ii)[1:] if ii >= 10 else signs[0] + str(ii)

        if pid in SEQ_PUZZLES:
            result = np.abs(answer_values - pred_ans).sum(axis=1) == 0
        elif pid in [32, 69, 82, 84, 95, 98, 51, 66, 44, 68]:
            result = [pred_ans[i] == answer[i] for i in range(len(pred_ans))]
        else:
            try:
                #NOTE : 근사한 값과 answer의 type(torch or numpyarray) 은 무조건 같아야함!!!
                result = (
                    np.abs(opts.astype("float")-pred_ans.unsqueeze(1).numpy()).argmin(axis=1)
                    == answer
                    )
            except:
                result = [pred_ans[i] == answer[i] for i in range(len(pred_ans))]
                print("error!!")
                pdb.set_trace()

        return np.array(result)

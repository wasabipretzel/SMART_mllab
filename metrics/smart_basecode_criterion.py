#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import read_dataset_info

class Criterion:
    def __init__(self):
        super(Criterion, self).__init__()
        self.monolithic = True  # just one classifier

        #option, a, av 같이 받아와야함
        self.puzzles = read_dataset_info("/data/SMART101-release-v1/puzzle_type_info.csv")

    def compute_metrics(self, pred):
        """_summary_
            dataset collator에 "labels" key가 있어야 eval_loop이후 결과가 여기로 들어옴
        Args:
            pred (EvalPrediction): include pred.predictions (pred value), pred.label_ids(answer)
                pred.label_ids : [[test_samples, 10] -> tensor
                pred.predictions : [test_samples, 257] (max_val + 1) ->  tensor
        """
        #b_options : [test_samples, 5] , b_answer_labels : [test_samples]
        b_options, b_answer_labels, b_pids = pred.inputs[0], pred.inputs[1], pred.inputs[2]

        # print(pred.inputs)
        tot_samples_num = pred.predictions.shape[0]
        pred_max = F.softmax(torch.tensor(pred.predictions), dim=1).argmax(dim=1).cpu() #tensor [samples]
        answer_value = torch.tensor(pred.label_ids[:,0]) #tensor [samples]
        s_acc = (pred_max == answer_value).float().sum() / tot_samples_num #NOTE 전체 수로 나눠줘야함
        #approximate to option (return array bool)
        opt_approximate = self.get_option_sel_acc(pred_max, b_options, b_answer_labels, answer_value, -1)

        #s_acc/o_acc per puzzle id
        puzzle_acc = {}
        for t in list(set(b_pids)):
            puzzle_acc[str(t)] = [
                (pred_max == answer_value)[b_pids == t].sum(),
                opt_approximate[b_pids == t].sum(),
                (b_pids == t).sum()
            ]
        
        to_int = lambda x: np.array(list(x)).astype("int")
        cls_mean = lambda x, idx, pids: np.array([x[int(ii)] for ii in idx]).sum() / len(
            set(to_int(idx)).intersection(set(to_int(pids)))
        )
        acc_list = np.zeros(101+1)
        opt_acc_list = np.zeros(101+1)
        for puzzle_id in puzzle_acc.keys():
            acc = 100.0 * puzzle_acc[puzzle_id][0] / puzzle_acc[puzzle_id][2]
            oacc = 100.0 * puzzle_acc[puzzle_id][1] / puzzle_acc[puzzle_id][2]
            acc_list[int(puzzle_id)] = acc
            opt_acc_list[int(puzzle_id)] = oacc
        #print acc, opt_acc by puzzle id
        for t in range(1, 101+1):
            print("%d opt_acc(%%)=%0.2f acc(%%)=%0.2f" % (t, opt_acc_list[t], acc_list[t]), end="\t")
            if t % 5 == 0:
                print("\n")
        print("\n\n")
        class_avg_perf = {}
        classes = ["counting", "math", "logic", "path", "algebra", "measure", "spatial", "pattern"]
        print(classes)
        for each_class in classes:
            idx_list = self.puzzles[each_class]
            class_avg_perf[each_class] = (
                cls_mean(acc_list, idx_list, list(puzzle_acc.keys())),
                cls_mean(opt_acc_list, idx_list, list(puzzle_acc.keys())),
            )
            print("%0.1f/%0.1f & " % (class_avg_perf[each_class][0], class_avg_perf[each_class][1]), end=" ")
        print("\n\n")

        result = {
            "S_acc" : s_acc*100,
            "O_acc" : 100*opt_approximate.sum() / tot_samples_num
        }
        #result에 class별 s_acc / o_acc append 혹은 update 
        for each_class in classes:
            result[f"{each_class}_acc"] = class_avg_perf[each_class][0]
            result[f"{each_class}_oacc"] = class_avg_perf[each_class][1]

        return result

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

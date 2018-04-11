import numpy as np



def get_ans_pointers(yp_mat):
    ans_index_mat = []
    for i, yi in enumerate(yp_mat):
        tmp = []
        tmp_idx = []
        end_index = len(yi[0])-1
        for j, yij in enumerate(yi):
            max_i = np.argmax(yij)
            if max_i == end_index or j == len(yi) - 1:
                tmp_idx.append(end_index)
                tmp.append([0, yij[end_index]])
                break
            else:
                tmp_idx.append(max_i)
                tmp.append([yij[max_i], yij[end_index]])
        print('tmp', tmp)
        one_ans = []
        pre_vec = []
        mul = 1
        for item in tmp:
            mul *= item[0]
            pre_vec.append(mul)

        for i in range(0, len(tmp) - 1):
            end_prob = tmp[i][1]
            cont_prob = 0
            for j in range(i, len(tmp) - 1):
                cont_prob += pre_vec[j] * tmp[j + 1][1]
            print(end_prob, cont_prob)
            if end_prob >= cont_prob:
                one_ans.append(end_index)
                print('end', end_index)
                break
            else:
                one_ans.append(tmp_idx[i])

            for j in range(i, len(pre_vec)):
                pre_vec[j] /= tmp[i][0]

        if len(one_ans) == 0 or one_ans[-1] != end_index:
            one_ans.append(end_index)
            print(one_ans)
        ans_index_mat.append(one_ans[:])



    return ans_index_mat  # [N, JA + 1]



a = [
    [[0.4, 0.2, 0.1, 0.25, 0.05],
     [0.1, 0.1, 0.6, 0.15, 0.05],
     [0.05, 0.45, 0.2, 0.05, 0.25],
     [0.2, 0.2, 0.25, 0.25, 0.1]
    ],
     [[0.4, 0.2, 0.1, 0.25, 0.05],
     [0.1, 0.1, 0.1, 0.15, 0.55],
     [0.05, 0.45, 0.2, 0.05, 0.25],
     [0.2, 0.2, 0.25, 0.25, 0.1]
    ]
    ]

ans_mat = get_ans_pointers(a)
print(ans_mat)
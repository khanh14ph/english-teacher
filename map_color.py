def map_color(score_list, start_index=1) -> dict:
    list_result = list()
    # print(score_list)
    for word in score_list:
        # right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score
        for right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score in word:
            print(right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score)
            if right_phoneme == model_predict_phoneme:
                final_score = max(right_phoneme_score, predict_score)
                if final_score >= 0.1:
                    tag = "green"
                else:
                    tag = '#ff9f1c'
            else:
                final_score = right_phoneme_score
                if final_score <= 0.05:
                    tag = 'red'
                else:
                    tag = '#ff9f1c'
            list_result.append((right_phoneme, model_predict_phoneme, final_score, tag))
        list_result.append((" "," ", 1, 'black'))

    if len(list_result) > 0:
        list_result.pop()
    return list_result
def map_color(score_list, start_index=1) -> dict:
    list_result = list()
    print(score_list)
    for word in score_list:
        # right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score
        for right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score in word:
            if right_phoneme == model_predict_phoneme:
                if right_phoneme_score >= 0.4:
                    tag = "right"
                else:
                    tag = 'neutral'
            else:
                if right_phoneme_score <= 0.1:
                    tag = 'wrong'
                else:
                    tag = 'neutral' # 'neutral'
            list_result.append((right_phoneme, tag))
        list_result.append((" ", 'normal'))
    if len(list_result) > 0:
        list_result.pop()
    return list_result
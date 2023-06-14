def map_color(score_list, start_index=1) -> dict:
    list_result = list()
    for word in score_list:
        for right_phoneme, model_predict, score in word:
            if right_phoneme == model_predict:
                tag = "right"
            else:
                tag = 'wrong' # 'neutral'
            list_result.append((right_phoneme, tag))
        list_result.append((" ", 'normal'))
    return list_result
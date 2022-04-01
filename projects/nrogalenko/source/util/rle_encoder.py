# convert 0,0,0,0,0 to 5_0
def rle_encode(data_str):
    current_seq = ""
    prev_seq = ""
    encoding = ""
    count = 1
    # remove header
    data = data_str[(str(data_str).find(',')):] + ","
    if not data:
        return ''
    for char in data:
        if char == ',':
            if current_seq == prev_seq:
                count += 1
            else:
                if prev_seq:
                    if count > 1:
                        encoding += str(count) + "_" + prev_seq + ","
                    else:
                        encoding += prev_seq + ","
                count = 1
            prev_seq = current_seq
            current_seq = ""
            continue
        current_seq += char
    if count > 1:
        encoding += str(count) + "_" + prev_seq + ","
    else:
        encoding += prev_seq + ","
    return data_str[:(str(data_str).find(',') + 1)] + encoding[:-1]


def rle_decode(text):
    freq_str = text[(text.find(',') + 1):]
    expanded_text_list = []
    # decoding rle
    for fr_info in freq_str.split(','):
        if "_" in fr_info:
            amount = int(fr_info.split("_")[0])
            fr = fr_info.split("_")[1]
            for i in range(0, amount):
                expanded_text_list.append(int(fr))
        else:
            expanded_text_list.append(int(fr_info))
    return expanded_text_list

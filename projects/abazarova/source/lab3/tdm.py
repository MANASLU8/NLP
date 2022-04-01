def tdm(tdm_file, dictionary, doc_count):
    # building and writing to file term document matrix
    ftdm = open(tdm_file, "w")
    header_string = ""
    for key in dictionary.keys():
        if "," in key:
            header_string += ",\"" + key + "\""
        else:
            header_string += "," + key
    ftdm.write(header_string)
    ftdm.write("\n")
    # add total docs containing token number
    ftdm.write('total_docs,' + run_length_encoding(','.join([str(len(dictionary[k])) for k in dictionary])))
    ftdm.write("\n")
    for i in range(1, doc_count+1):
        doc = str(i)
        for key in dictionary:
            if i in dictionary[key]:
                doc += "," + str(dictionary[key][i])
            else:
                doc += ",0"
        ftdm.write(run_length_encoding(doc))
        ftdm.write("\n")
        # print(i)
    ftdm.close()


def run_length_encoding(data_str):
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


def run_length_decoding(text):
    compressed = text[(text.find(',') + 1):]
    expanded = []
    # decoding rle
    for code in compressed.split(','):
        if "_" in code:
            amount = int(code.split("_")[0])
            numb = code.split("_")[1]
            for i in range(0, amount):
                expanded.append(int(numb))
        else:
            expanded.append(int(code))
    return expanded

a = (1, 2), (3, 4), (1, 3), (2, 3), (3, 1), (3, 2)


def my_map(key, values):
    temp_list = []
    for x in values:
        temp_list.append((x[0], 1))
    return temp_list


# print(my_map('Q1', a))

def my_reducer(intermediates):
    temp_list = []
    for key in dict(intermediates).keys():
        sumv = 0
        for pair in intermediates:
            if pair[0] == key:
                sumv += 1
        temp_list.append((key, sumv))
    return temp_list


# print(my_reducer(my_map('Q1', a)))




b = ('Doc-1', 'The map function that transforms, filters, or selects input data'), (
    'Doc-2', 'The reduce function that aggregates, combines, or collections results'), (
        'Doc-3', 'The map function and reduce function are invoked in sequence')

def map_index(doc):
    temp_dict = {}
    for key, words in doc:
        for word in words.replace(',', '').split():
            if word not in temp_dict.keys():
                temp_dict[word] = []
            temp_dict[word].append((key, 1))
    return temp_dict


def reduce_index(intermediate):
    temp_list = []
    for key, listOfValues in intermediate.items():
        doc_list = []
        for doc in dict(listOfValues).keys():
            sum = 0
            for values in listOfValues:
                if values[0] == doc:
                    sum += 1
            doc_list.append((doc, sum))
        temp_list.append((key, doc_list))
    return temp_list


# print(map_index(b))
# print(reduce_index(map_index(b)))


c = ('Doc-1', 'The map function that transforms, filters, or selects input data')


def map_index2(key, values):
    list = []
    for word in values.replace(',', '').split():
        list.append((word, (key, 1)))
    return list



def reduce_index2(key, values):
    temp_list = []
    temp_document_list = []
    for value in values:
        temp_document_list.append(value[0])
    document_set = set(temp_document_list)
    for doc in document_set:
        sumValue = 0
        for value in values:
            if value[0] == doc:
                sumValue += 1
        temp_list.append((doc, sumValue))
    return (key, temp_list)


print(reduce_index2('The', [('Doc-1', 1), ('Doc-2', 1), ('Doc-1', 1),]))





# print(map_index2(c[0], c[1]))

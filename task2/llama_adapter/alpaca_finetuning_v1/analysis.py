import json
from llama import Tokenizer
tokenizer = Tokenizer("llama/weights_7b/tokenizer.model")

data = json.load(open("finetuning_modified_content.json"))

string_length = []
string_lengths1 = []
string_lengths2 = []
string_lengths3 = []
new_string_lengths = []

new_data = []

for i in range(len(data)):   
    # new_inputs = data[i]["input"].replace("\n   ", " ").replace("input : [", "[")
    # new_data.append({"input": new_inputs, "instruction": data[i]["instruction"], "output": data[i]["output"]})
    l1 = tokenizer.encode(data[i]["output"], bos=True, eos=False)
    l2 = tokenizer.encode(data[i]["input"], bos=True, eos=False)
    l3 = tokenizer.encode(data[i]["instruction"], bos=True, eos=False)
    string_lengths1.append(len(l1))
    string_lengths2.append(len(l2))
    string_lengths3.append(len(l3))
    string_length.append(len(l1) + len(l2) + len(l3))
    if len(l1) + len(l2) + len(l3) < 320:
        new_data.append(data[i])
        new_string_lengths.append(len(l1) + len(l2) + len(l3))

    
print(max(string_lengths1))
print(max(string_lengths2))
print(max(string_lengths3))
print(max(string_length))
print(sum(string_lengths1) / len(string_lengths1))
print(sum(string_lengths2) / len(string_lengths2))
print(sum(string_lengths3) / len(string_lengths3))
print(sum(string_length) / len(string_length))



# print(max(string_lengths))
# print(sum(string_lengths) / len(string_lengths))

print(len(new_data))
print(sum(new_string_lengths) / len(new_string_lengths))

json.dump(new_data, open("300_old_content.json", "w"))
# print(max(string_lengths))
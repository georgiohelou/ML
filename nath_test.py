import pickle

# with open(r"abstracts.txt", 'r') as fp:
#     x = len(fp.readlines())
#     print('Total lines:', x)  # 8

a_file = open("abstracts_data.pkl", "rb")
output = pickle.load(a_file)

print("len(output) : ", len(output))
print(output[2908495138])

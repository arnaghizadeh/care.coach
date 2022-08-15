import matplotlib.pyplot as plt

from FinalSubmission.fileIO import get_file
labels = {'sentimental': 0, 'afraid': 1, 'proud': 2, 'faithful': 3, 'terrified': 4, 'joyful': 5,
          'angry': 6, 'sad': 7, 'jealous': 8, 'grateful': 9, 'prepared': 10, 'embarrassed': 11, 'excited': 12,
          'annoyed': 13, 'lonely': 14, 'ashamed': 15, 'guilty': 16, 'surprised': 17, 'nostalgic': 18,
          'confident': 19, 'furious': 20, 'disappointed': 21, 'caring': 22, 'trusting': 23, 'disgusted': 24,
          'anticipating': 25, 'anxious': 26, 'hopeful': 27, 'content': 28, 'impressed': 29, 'apprehensive': 30,
          'devastated': 31}

prompt_train, response_train, labels_train, prompt_val, response_val, labels_val = get_file()
D_train = {}

max_val = -1
max_label = -1
for i in labels_train:
    if i not in D_train:
        D_train[i] = 1
    else:
        D_train[i] += 1
    val = D_train[i]
    if val> max_val:
        max_val = val
        max_label = i

print(max_val,max_label,"out of:", len(labels_train))
acc = 0
for i in labels_val:
    if max_label == i:
        acc+=1
print("Correct for ZeroR is:", acc, " out of:", len(labels_val), "%"+str(acc/len(labels_val)*100))

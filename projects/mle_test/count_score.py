import os
def count_score(dir_path,test_path):
    score = 0
    num = 0
    for files in os.listdir(dir_path):
        file_path = os.path.join(dir_path, files)
        if os.path.isfile(file_path) and files.endswith('.txt'):
            with open(file_path, 'r') as f:
                line = f.read()
                try:
                    score += float(line.strip())
                except ValueError:
                    continue
    for competition_dir in os.listdir(test_path):
        if os.path.isdir(os.path.join(test_path, competition_dir)):
            num += 1
    print(f"Total competitions: {num}")
    return score/num
if __name__ == "__main__":
    dir_path = "/home/u-longyy/ms-agent/score_set_with_search_deepseek"
    test_path = "/home/u-longyy/MLE-Dojo/data/prepared"
    total_score = count_score(dir_path, test_path)
    print(f"Total Score: {total_score}")
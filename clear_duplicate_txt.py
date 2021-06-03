import sys

txt_dir = sys.argv[1]

# clear duplicate lines in our broken images text file
def clear_duplicate():
    s = set()
    with open(txt_dir, "r") as file_object:
        l = file_object.readlines()
        for path in l:
            s.add(path)
    with open(txt_dir, "w") as file_object:
        for path in s:
            file_object.write(path)

if __name__ == "__main__":
    clear_duplicate()


open_filename = './env.txt'
save_filename = './requirements.txt'

print(f"open from: {open_filename}")
print(f"save to: {save_filename}")

def split_line(line):
    flag = 0
    for v in line.split(" "):
        if flag == 1:
            v = v.split(".")[0]
            v = v.split("\n")[0]
            if (v == "*"):
                return 0
            return v
        if(v == "import"):
            flag=1

if __name__ == '__main__':
    with open(open_filename) as f:
        data = f.readlines()
        libs = []
        for line in data:
            line = split_line(line)
            if (line == 0) | (line == None):
                continue
            if len(line) == 1:
                continue
            if line[0].isupper() | (',' in line):
                continue
            libs.append(line + '\n')
    libs = list(set(libs))
    with open(save_filename, 'w') as f:
        f.writelines(libs)
    print("write finish.")

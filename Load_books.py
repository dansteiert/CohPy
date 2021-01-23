import json, os,wget


def load_gutenberg(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data
    
    
def download_files(data, path_for_download):
    if not os.path.isdir(path_for_download):
        os.mkdir(path_for_download)
    num_books = len(data["books"])//20
    print("start loading books (5% steps):")
    for index, book in enumerate(data["books"]):
        if os.path.isfile(os.path.join(path_for_download, "%s.txt" % book["id"])):
            continue
        if index % num_books == 0 and index !=0:
            print("#", end="")
        
        for i in book["files"]:
            if ".txt" == i["url"][-4:]:
                try:
                    wget.download(i["url"], os.path.join(path_for_download, "%s.txt" % book["id"]))
                except:
                    print(i["url"], "not found")
                break

path = os.path.join(os.getcwd(), "data", "Gutenberg", "data.json")
path_for_download = os.path.join(os.getcwd(), "data", "Gutenberg", "txt_files")

data = load_gutenberg(path)
download_files(data=data, path_for_download=path_for_download)
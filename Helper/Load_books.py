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
        if index % num_books == 0 and index !=0:
            print("#", end="")
        if os.path.isfile(os.path.join(path_for_download, "%s.txt" % book["id"])):
            continue

        
        for i in book["files"]:
            if ".txt" == i["url"][-4:]:
                try:
                    wget.download(i["url"], os.path.join(path_for_download, "%s.txt" % book["id"]))
                except:
                    print(i["url"], "not found")
                break


def search_title(data, title):
    book_check = [i for i in data["books"] if title in i["title"].lower()]
    for i in book_check:
        print(i)

def find_author(data, author):
    author_check = [i for i in data["books"] for j in i["authors"] if author in j["name"]]
    id_list = []
    for i in author_check:
        id_list.append(i["id"])
        print(i)
    return id_list
        



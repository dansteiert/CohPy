import json, os,wget

def load_gutenberg(path):
    """
    Load the Gutenberg Metadata file from the given path
    :param path: str, path to gutenberg metadata file
    :return: dict, {books: list[{the single gutenberg entries}]}
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data
    
    
def download_files(data, path_for_download):
    """
    Download all books from the gutenberg project, and save them at the designated folder in path_for_download.
    Each file is written out as gutenberg_id.txt, to ensure uniqueness.
    If not the whole project needs to be downloaded, the data variable (the gutenberg meta data file) needs to be
    reduced to just those entries.
    The progress can be stopped at any point and resumed later on with out saving anything.
    A non German VPN connection is currently (February 2021) necessary to access the Gutenberg Project.
    :param data: dict, {books: list[{the single gutenberg entries}]} gutenberg metadata - retrieved from load gutenberg
    :param path_for_download:
    :return:
    """
    if not os.path.isdir(path_for_download):
        os.mkdir(path_for_download)
    num_books = len(data["books"])//20
    print("start loading books (5% steps):", end="")
    for index, book in enumerate(data["books"]):
        if index % num_books == 0 and index !=0:
            print("#", end="")
        # check if the file already exists
        if os.path.isfile(os.path.join(path_for_download, "%s.txt" % book["id"])):
            continue
        # search for a .txt file and download it, if possible (only a single .txt file is retrieved)
        for i in book["files"]:
            if ".txt" == i["url"][-4:]:
                try:
                    wget.download(i["url"], os.path.join(path_for_download, "%s.txt" % book["id"]))
                except:
                    print(i["url"], "not found")
                break


def search_title(data, title):
    """
    A function to search for specific titles within the corpus, using the metadata, their metadata entries are returned
    :param data: dict, {books: list[{the single gutenberg entries}]} gutenberg metadata - retrieved from load gutenberg
    :param title: str, the title of the book in question. both book title and this string are cast to lower case for the matching.
    :return: None, the whole metadata for the book found is printed out
    """
    book_check = [i for i in data["books"] if title.lower() in i["title"].lower()]
    for i in book_check:
        print(i)

def find_author(data, author):
    """
    A function to search for a specific author, by full-, first or lastname. Cases are regarded
    :param data: dict, {books: list[{the single gutenberg entries}]} gutenberg metadata - retrieved from load gutenberg
    :param author: str, name of the author
    :return: returned is the gutenberg_id list, as well as the whole metadata for those books are printed out for fine
    selection purposes
    """
    author_check = [i for i in data["books"] for j in i["authors"] if author in j["name"]]
    id_list = []
    for i in author_check:
        id_list.append(i["id"])
        print(i)
    return id_list
        



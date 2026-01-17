import wget

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

wget.download(url, "data/tiny_shakespeare.txt")

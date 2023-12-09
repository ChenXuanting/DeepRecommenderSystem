from dataset.download_data import download_amazon_review_data

review_name = "reviews_Electronics_10"
downloaded_data = download_amazon_review_data(review_name)
print(downloaded_data)
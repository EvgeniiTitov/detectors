import os
import cv2
import argparse
import requests
import time


URL_ENDPOINT = r"https://api.cognitive.microsoft.com/bing/v7.0/images/search"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Webscraping with Bing API")
    parser.add_argument("-q", "--query", required=True, help="Search query")
    parser.add_argument("-k", "--key", required=True, help="API key")
    parser.add_argument("--save_path", default=r"C:\Users\Evgenii\Downloads\scraping",
                        help="Path to where results will be saved")
    parser.add_argument("--max_results", type=int, default=200,
                        help="Total number of images that will be downloaded")
    parser.add_argument("--group_size", type=int, default=50, help="Offset size")
    arguments = parser.parse_args()

    return arguments


def attempt_reading_image(path: str) -> bool:
    image = cv2.imread(path)
    if image is None:
        return False
    return True


def download_batch_of_images(results: dict, save_path: str, total: int) -> int:
    for v in results["value"]:
        try:
            print("Fetching:", v["contentUrl"])
            result = requests.get(v["contentUrl"], timeout=30)

            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            path = os.path.sep.join([save_path, "{}{}".format(str(total).zfill(8), ext)])
            file = open(path, "wb")
            file.write(result.content)  # .content gives access to binary data
            file.close()
        except Exception as e:
            print(f"Exception raised: {e}. Skipping: {v['contentUrl']}")
            continue
        # Check if the image has been successfully downloaded and can be opened
        success = attempt_reading_image(path)
        if not success:
            print("Failed image reading test. Deleting:", path)
            os.remove(path)
        total += 1

    return total


def execute_request(headers: dict, params: dict, save_path: str, max_results: int, group_size: int) -> None:
    search = requests.get(URL_ENDPOINT, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    esimated_nb_results = min(results["totalEstimatedMatches"], max_results)
    print(f"\nFound {esimated_nb_results} results for query: {params['q']}")
    time.sleep(2)

    # Traverse over estimated number of results with the GROUP SIZE step
    total_downloaded = 0
    for offset in range(0, esimated_nb_results, group_size):
        # using the current offset update the search parameters, make a request to fetch the results
        print(f"\nRequesting group {offset}-{offset + group_size} of {esimated_nb_results}...")
        params["offset"] = offset
        search = requests.get(URL_ENDPOINT, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print(f"\nSaving images for group {offset}-{offset + group_size} of {esimated_nb_results}...")
        total_downloaded = download_batch_of_images(results, save_path, total_downloaded)

    return


def main():
    args = parse_arguments()
    query = args.query
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    assert 0 < args.max_results <= 1000, "Wrong value of max results. Expected 1 - 1000"
    assert 0 < args.group_size < args.max_results, "Wrong value of group size. Expected: 1 - max_results"

    print("\nSEARCHING IMAGES FOR QUERY:", query)
    headers = {"Ocp-Apim-Subscription-Key": args.key}
    params = {"q": query, "offset": 0, "count": args.group_size}
    execute_request(headers, params, save_path, args.max_results, args.group_size)


if __name__ == "__main__":
    main()

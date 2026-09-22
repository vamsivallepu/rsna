"""Download large Kaggle kernel outputs using the client's resumable streamer."""
import argparse
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(args.kernel)
    token = None
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        api._set_paging(request, 20, token)
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            path = (args.output / item.file_name).resolve()
            if not path.is_relative_to(args.output.resolve()):
                raise ValueError('Output path escapes its directory')
            # kernels_output uses response.content; this API streams and safely resumes.
            with requests.get(item.url, stream=True, timeout=(30, 300)) as download:
                download.raise_for_status()
                api.download_file(download, str(path), None, quiet=True, resume=True)
            print(f'Downloaded {path.name}: {path.stat().st_size} bytes', flush=True)
        token = response.next_page_token
        if not token:
            return


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Signed download URLs must not leak through HTTP exception text.
        print(f'Download failed: {type(error).__name__}', flush=True)
        raise SystemExit(1)

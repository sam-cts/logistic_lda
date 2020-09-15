mkdir -p data
cd data

# download data
wget --header 'Host: uc146cdf5b57b964b4fdc7823962.dl.dropboxusercontent.com'\
     --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'\
     --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'\
     --header 'Accept-Language: en-US,en;q=0.5'\
     --referer 'https://www.dropbox.com/'\
     --header 'DNT: 1' \
     --header 'Upgrade-Insecure-Requests: 1' 'https://uc146cdf5b57b964b4fdc7823962.dl.dropboxusercontent.com/cd/0/get/A_ZEsR2KMrd4uRA0QA0VfuejKUvFxkPrTmlaVSEfavuQ244r-cjXQ_F-gZdIgqSLoWmk6NarUM2dnIBYhO0Kl84ns43E_ih_XaFHC9Z5YWXXpSeHF-JjKpjau8tt6JmuPGc/file#' \
     --output-document 'data_20200823.pkl'
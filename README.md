# HAL
**Simple, customizable python script to send you a mail summarizing all mails of the last 24hours, as well as adding user-defined labels**

## Features
* **Simple** A unique python file. Few arguments. All settings stored in a single `.toml` file.
* **Time efficient** By default each mail is summarized as a max 5 sentence summary, then as a 1 sentence summary.
* **Customizable** Prompts used to summarize the mails are accessible in the toml settings file and can be tailored to any user or team. Works on any language, on any topic.
* **Lightweight and robust** Few dependencies, relying on standards (simple HTML, no special fonts, IMAP/SMTP). HAL.py can be used on any standard mail server. (Mostly tested on GMail so far).
* **By a friendly developer** For anything at all, just open up an issue!
* **User-defined labels support** Each inbox mail can be assigned a user defined label. The labels are either GMail flavored labels (non standard) or simply mentioned in the summary mail (robust).
* **LLM Agnostic** [LiteLLM](https://docs.litellm.ai/docs/providers/) is used to enable calling any LLM with only 3 modifications to the settings (API KEY, name of the LLM, and specifying the price). For example OpenAI, Mistral, Hugginface, Claude, ...
* **Cost controlled** Detailed cost are written in the summary mail. In addition, several failsafes are in place to stop bad actors from costing you anything unexpected.
* **Obedient** With `--interactive`: asks for confirmation before doing anything important.
* **Reasonably secure**
    1. Any link found in the Incoming mail is redacted before sending it to the LLM for summary. This reduces costs but more importantly mitigates the risk of a malicious email using prompt injection techniques (or not yet discovered techniques!) to exfiltrate information etc.
    2. Attachment names are accessible to the LLM but not the content.
    3. HAL.py will ask for confirmation if receiving an unexpectely large number of mails (see `--n_mail_limit`).
    4. HAL.py will ask for confirmation if receiving an unexpectedly large email (see `--tkn_warn_limit`)


## How to
**Note: this code was written and tested on python 3.11.7**

0. If you're using GMail, follow [this guide](https://www.makeuseof.com/gmail-use-imap/) to generate an API key that you'll use as 'IMAP password'.
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Configuration: duplicate the file `default_settings.toml` to something like 'your_settings.toml' and edit its content to suit your needs.
3. To see how to use HAL.py, read the section `Arguments` below.
* Example run: `python HAL.py --inbox_mail johndoe@gmail.com --inbox_imap_password "YOUR_IMAP_PASSWORD" --settings_file "your_settings.toml"`
* For people uncomfortable with the command line: edit the `run.sh` file, `chmod +x` then just double click on it every morning.
* For scheduled run, you can use cron on UNIX (OSX/Linux).
* *Note: The labels can only be turned into proper "webmail labels" if the remote server is gmail. Otherwise the labels are just strings mentioned in the summary.*

<details><summary><strong>Usage</strong></summary><pre><code>
inbox_mail: str
    the mail you want to login to
inbox_imap_password: str
    the IMAP password of inbox_mail
settings_file: str, default "default_settings.toml"
    path to the settings file to use.
dont_labellize: bool, default False
    if True, will not add labels to the remote server. Used for
    debugging.
dont_send_summary: bool, default False
    if True, will not send the summary to the recipients. Used for
    debugging.
disable_labels_entirely: bool, default False
    if True, will not ask the LLM for labels.
interactive: bool, default True
    ask for confirmation at every important step. If you answer
    anything else than "y" the script will completely stop. Mostly
    for debugging.
verbose: bool, default True
use_cache: bool, default False
    if True, will import joblib.Memory and cache the LLM calls.
    Normally only relevant for debugging.
tkn_warn_limit: int, default 3000
    if any mail contains more tokens than this, the script will
    ask for confirmation
total_cost_limit: int, default 1
    if the expected cost is larger than $1, the script will
    ask for confirmation
n_mail_limit: int, default 20
    if the number of mail in the inbox since yesterday is larger than
    this, the script will ask for confirmation
detailed_price: bool, default False
    if True, will mention the detailed breakdown of the price after
    each summary. If False, the details can be found as commented
    HTML in the summary mail.
</code></pre></details>

<details><summary><strong>Default settings for the .toml file</strong></summary><pre><code>
language = "fr"
# any of "fr" or "en". This is used for the strings that appear
# in the summary. Also the dates mentionned will use DD/MM/YYYY in
# "fr" but MM/DD/YYYY in "en"
imap_domain = "imap.gmail.com"
imap_port = 993
smtp_domain = "smtp.gmail.com"
smtp_port = 587
summary_recipients = [ "some_mail@example.org", "other_mail@anotherexample.com",]
available_labels = [ "Contract", "Threats", "Spam", "Newsletter", "From a human", "Money", "Miscellaneous", "Urgent", "Others", ]
labellizer_prompt = """
Tu es mon meilleur assistant. J'ai besoin que tu sélectionne un seul label qui correspondent le mieux à un mail dont je te donne le résumé. Cette tâche est cruciale pour mon entreprise et si tu l'accomplis en respectant les règles je te donnerai une augmentation importante !
Voici les règles que tu dois impérativement respecter :
    - Il est absolument crucial que ta réponse ne contienne qu'un seul label et rien d'autre.
    - Tu n'as pas le droit de mentionner de label qui ne soit pas dans la liste.
    - N'ajoute ni majuscule ni ponctuation au label. Répond simplement le label de la liste sans être polie ni accuser réception de ces règles.
    - Ne t'adresse pas à moi, si tu n'arrives pas à catégoriser en respectant ces simples règles tu seras viré.
Voici un exemple de réponse valide:
```
[LABEL_EXAMPLE]
```
Voici la liste des labels parmi lesquels tu dois choisir: [[LABEL_LIST]]
Je te remercie, nous comptons tous sur toi !
"""
summarizer_prompt = '''
Tu es mon meilleur assistant. J'ai besoin que tu me résumes un email en quelques lignes en suivant des règles. Cette tâche est cruciale pour mon entreprise et si tu l'accomplis en respectant les règles je te donnerai une augmentation importante !
Voici les règles que tu dois impérativement respecter :
    - Ecrit ton résumé en français.
    - Ta réponse ne doit contenir que le résumé du mail, n'accuse pas réception de mes instructions.
    - Ton résumé doit préciser pourquoi la personne nous contacte et ce qu'elle nous demande.
    - Ton résumé doit faire un maximum de 5 phrases. Et chaque phrase doit être sur une ligne (donc un <br> doit separer chaque ligne de ton résumé)
    - Les personnes physiques ou morales mentionnés dans ton résumé doient apparaître en gras.
    - Le format de ton résumé doit être directement du HTML mais n'utilise pas de délimiteurs comme '``` html': répond <b>directement</b> du HTML. Ta réponse sera intégrée dans un <p> donc inutil de l'entourer de <p> ou de <html> <body> etc.
    - Si le mail s'adresse à une personne en particulier, ton résumé doit le préciser. S'il y a plusieurs destinataires, ou que des gens sont en CC, soit précis dans ton résumé pour que l'on sache bien à qui quoi est demandé etc.
    - Ne précise pas de choses sans rapport avec le contenu du mail (par exemple la présence d'un lien pour de désinscription de newsletter).
    - Si le mail est vide ou incomplet, ton résumé doit l'expliquer en restant très concie, inutile de détailler que tu n'as pu accomplir ta tache et tu seras quand même récompensé.
Je te remercie, nous comptons sur toi !
'''
short_summarizer_prompt = '''
Tu es mon meilleur assistant. Je te donne un résumé en html d'un mail en quelques phrases et tu dois me répondre immédiatement un condensé en <b>une seule phrase concise et courte</b> de ce résumé de mail. Ne cherche pas à être polie : le résumé doit être rapide à lire. Le format doit être directement du HTML, sans délimiteur comme '``` html' etc. Utilise <b>du gras</b> pour faire ressortir les éléments importants. Utilise la même langue que le résumé. Cette tâche est cruciale pour mon entreprise et si tu l'accomplis en respectant les règles je te donnerai une augmentation importante !
Je te remercie, nous comptons sur toi !
'''
llm_name = "openai/gpt-4-1106-preview"
# llm_name = "mistral/mistral-large-latest"
LLM_API_KEY = "YOUR_KEY"
# for gpt-4-1106-preview
[llm_price]
prompt = 0.01
completion = 0.03
unit = "dollar"
# for mistral-large-latest
# [llm_price]
# prompt = 0.008
# completion = 0.024
# unit = "euro"
</code></pre></details>


## FAQ
* **Why did you do this?** A friend works for a well known international company and I kept telling him to use AI to make better use of his time at work. To prove my point I coded HAL.py in a few hours.
* **Can HAL.py read my attachments?** HAL.py can access the attachments name but not their contents. Parly because reading the content would be a security risk, but also because it could make the cost of an email unpredictable: if a colleague sends you a short mail asking you to read a long attached PDF, you don't want HAL.py to read the whole PDF.
* **What happens if I receive millions of main in a spam attack?** HAL.py will ask for confirmation if more than `--n_mail_limit` mails are received.
* **What happpens if I receive a very very long email?** HAL.py will ask for confirmation if an email contains more tokens than `--tkn_warn_limit`.


## Credits
* OC and AJ, and the company that employs him.

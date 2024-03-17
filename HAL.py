import random
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os
import sys
import re
import toml
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from textwrap import dedent

import fire
import pyfiglet
import chardet
from email_validator import validate_email, EmailNotValidError
from tqdm import tqdm
from bs4 import BeautifulSoup
import ftfy
import tiktoken
from litellm import completion

# timestamp used to create 1 logging files per run
d = datetime.today()
timestamp_en = f"{d.day:02d}_{d.month:02d}_{d.year:04d}_at_{d.hour:02d}:{d.minute:02d}"
timestamp_fr = f"{d.day:02d}/{d.month:02d}/{d.year:04d} à {d.hour:02d}:{d.minute:02d}"

# create logger
Path("logs").mkdir(exist_ok=True)
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s"
)
file_handler = logging.FileHandler(
    f"logs/{timestamp_en}.txt",
    mode="a",
    encoding=None,
    delay=0,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)

# tokenizer to know the length of a prompt before sending it
# the tokenizer always used is the one from ChatGPT. This might not alway
# be accurate but will remain not too far from the truth
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def tokenize(text):
    "get token length of a string"
    return len(tokenizer.encode(text))


def p(s):
    "quick way to print and log to file at the same time"
    log.info(s)
    tqdm.write(s)


class HAL:
    VERSION = "1.0.0"

    def __init__(
        self,
        inbox_mail: str,
        inbox_imap_password: str,
        settings_file: str = "default_settings.toml",
        dont_labellize: bool = False,
        dont_send_summary: bool = False,
        disable_labels_entirely: bool = False,
        interactive: bool = True,
        verbose: bool = True,
        use_cache: bool = False,
        tkn_warn_limit: int = 3000,
        total_cost_limit: int = 1,
        n_mail_limit: int = 20,
        detailed_price: bool = False,
    ):
        """
        Parameters
        ----------
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
        """
        # show banner
        p(pyfiglet.figlet_format(f"\nHAL\nv{self.VERSION}"))

        # store arguments as attributes
        self.inbox_mail = inbox_mail
        self.inbox_imap_password = inbox_imap_password
        self.interactive = interactive
        self.verbose = verbose
        self.tkn_warn_limit = tkn_warn_limit
        self.total_cost_limit = total_cost_limit
        self.n_mail_limit = n_mail_limit
        self.dont_labellize = dont_labellize
        self.dont_send_summary = dont_send_summary
        self.disable_labels_entirely = disable_labels_entirely
        self.use_cache = use_cache
        self.detailed_price = detailed_price

        if use_cache:
            from joblib import Memory

            self.mem = Memory("cache", verbose=verbose)

        # load settings
        self.sf = Path(settings_file)
        assert self.sf.exists(), f"File not found: {settings_file}"
        with open(self.sf.absolute(), "r") as f:
            self.settings = toml.load(f)
        assert isinstance(
            self.settings, dict
        ), f"Invalid settings type: {self.settings}"
        expected_keys = [
            "language",
            "imap_domain",
            "imap_port",
            "smtp_domain",
            "smtp_port",
            "summary_recipients",
            "llm_name",
            "llm_price",
            "LLM_API_KEY",
            "summarizer_prompt",
            "short_summarizer_prompt",
        ]
        if not disable_labels_entirely:
            expected_keys += ["labellizer_prompt", "available_labels"]
        for key in expected_keys:
            assert key in self.settings, f"Key not found in settings: {key}"
            setattr(self, key, self.settings[key])
        for key in self.settings:
            if key not in expected_keys:
                raise Exception(f"Found unexpected key in settings: {key}")
        self.fiat_name = self.settings["llm_price"]["unit"]
        assert self.language in [
            "fr", "en"], f"Invalid language: {self.language}"

        # if the server is not gmail, disable remote labellizer
        if not disable_labels_entirely:
            if "gmail" not in self.smtp_domain or "gmail" not in self.imap_domain:
                if self.dont_labellize:
                    p(
                        "The server does not appear to be GMail, labels will not "
                        "be applied to the remote server and only be mentionned "
                        "in the summary."
                    )
                self.dont_labellize = True

        # set API key
        backend = self.llm_name.split("/")[0].upper()
        os.environ[f"{backend}_API_KEY"] = self.LLM_API_KEY

        # check that emails are present and valid
        try:
            validate_email(inbox_mail)
        except EmailNotValidError as e:
            raise Exception(
                f"The inbox_mail you entered appears to be invalid: {e}")
        for tm in self.summary_recipients:
            try:
                validate_email(tm)
            except EmailNotValidError as e:
                raise Exception(f"Invalid mail in target email ({tm}): {e}")

        # prepare the mail backend
        try:
            self.imap = imaplib.IMAP4_SSL(
                host=self.imap_domain, port=self.imap_port)
            assert self.imap.login(
                inbox_mail, self.inbox_imap_password
            ), "Invalid login"
        except Exception as err:
            raise Exception(f"Failed to initialize IMAP: '{err}'")

        try:
            self.smtp = smtplib.SMTP(self.smtp_domain, self.smtp_port)
            self.smtp.starttls()
            assert self.smtp.login(
                inbox_mail, self.inbox_imap_password
            ), "Failed smtp login"
        except Exception as err:
            raise Exception(f"Failed to initialize SMTP: '{err}'")

        # make sure the labels are ascii compliant
        if not self.disable_labels_entirely:
            self.available_labels = [
                lab.encode("ascii", "ignore").decode() for lab in self.available_labels
            ]

            # update the labellizer prompt
            assert (
                "[LABEL_LIST]" in self.labellizer_prompt
            ), "Missing '[LABEL_LIST]' in labellizer_prompt"
            assert (
                "[LABEL_EXAMPLE]" in self.labellizer_prompt
            ), "Missing '[LABEL_EXAMPLE]' in labellizer_prompt"
            self.labellizer_prompt = self.labellizer_prompt.replace(
                "[LABEL_LIST]", "'" + "','".join(self.available_labels) + "'"
            ).replace("[LABEL_EXAMPLE]", self.available_labels[0])

        # recap to the user
        p(f"## Mail adress who's inbox will be scanned: {inbox_mail}")
        if not disable_labels_entirely:
            p(f"## Available labels: {','.join(self.available_labels)}")
            if dont_labellize:
                p("## Labels will not be sent to the server. (DEBUG MODE)")
        p(f"## Recipients of the summary: {','.join(self.summary_recipients)}")
        if dont_send_summary:
            p("## Summary will not be sent to the recipients. (DEBUG MODE)")
        p(f"## LLM model to use: {self.llm_name}")
        if self.verbose:
            if not disable_labels_entirely:
                p("## Prompt to create labels:\n'''\n")
                p(self.labellizer_prompt)
                p("'''")
            p("## Prompt to summarize:\n'''\n")
            p(self.summarizer_prompt)
            p("'''")
            p("## Prompt to shorten the summary:\n'''\n")
            p(self.short_summarizer_prompt)
            p("'''")
        self.interact()

        # actual execution
        p("Fetching and parsing mail from yesterday.")
        self.fetch_yesterday_mail()
        self.parse_each_mail()

        if disable_labels_entirely:
            p("Summarize each mail.")
        else:
            p("Labellize and summarize each mail.")
        self.interact()
        self.process_each_mail()
        self.formating_summary_mail()

        p("Sending the summary.")
        self.interact()
        self.send_summary()

        p("Mail sent.")

        # all done
        self.exit()

    def interact(self, message="Do you confirm?"):
        """If the argument interactive is True then key steps of the
        program will require manual confirmation. Otherwise this does nothing.
        """
        if self.interactive:
            ans = input(f"{message} (y/n/debug)")
            if ans == "debug":
                breakpoint()
                return self.interactive("Continue?")
            elif ans != "y":
                raise Exception("Exit.")

    def fetch_yesterday_mail(self):
        "load the mail received since yesterday"
        self.imap.select("INBOX")
        state, ids = self.imap.uid(
            "search",
            None,
            "(SENTSINCE {date})".format(
                date=(datetime.now() - timedelta(1)).strftime("%d-%b-%Y")
            ),
        )
        assert state == "OK", f"Invalid response: {state}, {ids}"
        ids = ids[0].decode().split(" ")
        assert ids, "No mail received since yesterday"
        inbox_mails = [self._mailinfo(mid) for mid in ids]

        # filter out mail from HAL itself
        inbox_mails = [
            m for m in inbox_mails if not m["Subject"].startswith("HAL - ")]
        p(f"Found {len(ids)} mails since yesterday")
        if len(inbox_mails) > self.n_mail_limit:
            self.interact(
                f"Number of mail to process is {len(inbox_mails)} "
                f"which is above {self.n_mail_limit}. Do you "
                "confirm?"
            )
        self.inbox_mails = inbox_mails
        return

    def parse_each_mail(self):
        """for each mail, parse its content and metadata into an
        LLM friendly format"""
        for mail in self.inbox_mails:
            subj = mail["Subject"]
            date = " ".join(mail["Date"].split(" ")[:-2])
            sender = mail["From"]
            if "Cc" in mail:
                in_cc = mail["Cc"]
            else:
                in_cc = "None"
            if "Reply-To" in mail:
                rpt = mail["Reply-To"]
            else:
                rpt = "None"
            destin = mail["To"]

            # if too many cc or recipent, redact the middle
            if len(destin) > 100:
                destin = destin[:50] + " [TOO_MANY] " + destin[:-50:]
            if len(rpt) > 100:
                rpt = rpt[:50] + " [TOO_MANY] " + rpt[:-50:]
            if len(in_cc) > 100:
                in_cc = in_cc[:50] + " [TOO_MANY] " + in_cc[:-50:]

            # shorten subject if too long
            if len(subj) > 100:
                subj = subj[:100] + " [TOO LONG]"

            n_attach = len(mail["attachments"])

            # format the attachment names
            attach_list = []
            if n_attach:
                attach_list = mail["attachments"]

                # make sure that the name of the file is not longer than 15 characters
                # but keep the extension
                for i, a in enumerate(attach_list):
                    if "." not in a:
                        continue
                    if len(a) > 15:
                        ext = a.split(".")[-1]
                        a = f"{a[:13]}.{ext}"
                        attach_list[i] = a
            mail["attachments_list"] = attach_list

            # get the mail content
            # format the mail content as something easy to parse by the llm
            txt = [
                BeautifulSoup(dc, "html.parser").get_text()
                for dc in mail["decoded_content"]
            ]
            txt = [ftfy.fix_text(t) for t in txt]
            txt = "\n".join(txt)

            # remove the api key if by any chance it's in the mail
            txt = txt.replace(self.LLM_API_KEY, "[API_KEY_REDACTED]")

            # remove http links as they consume a lot of tokens
            txt = re.sub(r"(http|https)://[^\s]+", "[HTTP_LINK]", txt)

            # replace long spaces
            txt = txt.replace("\xa0", " ").replace("\xc2", " ")

            # always use the same newline, and restrict how many in a row
            txt = txt.replace("\r", "\n")
            txt = "\n".join([t.strip() for t in txt.splitlines()])

            # remove all newlines if many tokens
            if tokenize(txt) > 2000:
                txt = txt.replace("\n", "")

            content = dedent(f"""
            Subject: '{subj}'
            Date: '{date}'
            Sender: '{sender}'
            Recipients: '{destin}'
            CC: '{in_cc}'
            Reply-To: '{rpt}'""")
            if n_attach:
                content += f"\Attachments: '{', '.join(attach_list)}'"
            content += dedent("""
            Body:
            '''
            TXT
            '''
            """).replace("TXT", txt.strip())
            content = dedent(content).strip()

            # store in the dictionnary
            mail["ready_to_summarize"] = content
            try:
                parsed_date = mail["Date"]
                if "(" in parsed_date:
                    # detect timezone
                    tz = parsed_date.split("(")[1].split(")")[0]
                    parsed_date = parsed_date.replace(f"({tz})", "").strip()
                else:
                    tz = "UTC"
                parsed_date += " (UTC)"
                parsed_date = datetime.strptime(
                    parsed_date, "%a, %d %b %Y %H:%M:%S %z (UTC)"
                )
                if tz != "UTC":
                    parsed_date = parsed_date.astimezone(ZoneInfo(tz))
                if self.language == "fr":
                    mail["parsed_date"] = (
                        f"{parsed_date.day}/{parsed_date.month}/{parsed_date.year} à {parsed_date.hour}:{parsed_date.minute:02d}"
                    )
                elif self.language == "en":
                    mail["parsed_date"] = (
                        f"{parsed_date.month}/{parsed_date.day}/{parsed_date.year} at {parsed_date.hour}:{parsed_date.minute:02d}"
                    )
            except Exception as err:
                p(f"Error when converting date '{mail['Date']}': '{err}'")
                mail["parsed_date"] = mail["Date"]

    def process_each_mail(self):
        """for each mail, get the labels and the summary via the LLM."""
        total_fiat_cost = 0
        for mail in tqdm(self.inbox_mails, desc="Processing", unit="mail"):
            content = mail["ready_to_summarize"]
            p("\n\nMail to summarize:")
            p(content)
            tk = tokenize(content)
            p(f"Number of tokens: {tk}")

            # check that the run will not be unexpectedly expensive
            if tk >= self.tkn_warn_limit:
                self.interact(
                    f"Number of token " f"is above {self.tkn_warn_limit}.")

            # get summary from LLM
            p("Summarizing.")
            ans_summary = self._summarizer(
                content, prompt=self.summarizer_prompt)
            mess_summary = ans_summary["choices"][0]["message"]["content"]

            p("Shortening the summary.")
            short_ans_summary = self._summarizer(
                mess_summary, prompt=self.short_summarizer_prompt
            )
            short_mess_summary = short_ans_summary["choices"][0]["message"]["content"]

            # get labels from LLM
            if not self.disable_labels_entirely:
                p("Labellizing")
                ans_label = self._labelizer(short_mess_summary)
                mess_label = ans_label["choices"][0]["message"]["content"].strip(
                )
                if mess_label not in self.available_labels:
                    if self.interactive:
                        self.interact(
                            f"The LLM selected label '{mess_label}' which is not "
                            f"in {self.available_labels}"
                        )
                    else:
                        p(
                            f"The LLM selected label '{mess_label}' which is "
                            "not in {self.available_labels}"
                        )

            # get cost of both LLM calls
            input_tokens = ans_summary["usage"]["prompt_tokens"]
            output_tokens = ans_summary["usage"]["completion_tokens"]
            tkn_cost = input_tokens + output_tokens
            fiat_cost = (
                input_tokens / 1000 * self.llm_price["prompt"]
                + output_tokens / 1000 * self.llm_price["completion"]
            )
            sum_fiat_cost = fiat_cost
            # add the call from the short summary
            input_tokens += short_ans_summary["usage"]["prompt_tokens"]
            output_tokens += short_ans_summary["usage"]["completion_tokens"]
            if not self.disable_labels_entirely:
                input_tokens += ans_label["usage"]["prompt_tokens"]
                output_tokens += ans_label["usage"]["completion_tokens"]
            tkn_cost += input_tokens + output_tokens - tkn_cost
            fiat_cost += (
                input_tokens / 1000 * self.llm_price["prompt"]
                + output_tokens / 1000 * self.llm_price["completion"]
                - fiat_cost
            )
            label_fiat_cost = fiat_cost - sum_fiat_cost

            # show results
            p("\n###\nMail summary:")
            p(BeautifulSoup(mess_summary, "html.parser").get_text())
            p("\nShortened to:")
            p(BeautifulSoup(short_mess_summary, "html.parser").get_text())
            if not self.disable_labels_entirely:
                p(f"Found label: {mess_label}")
            p(f"Token cost for input: {input_tokens} and output {output_tokens}")
            p(f"Fiat cost in {self.fiat_name}: {round(fiat_cost, 5)}")
            p("###\n")
            # self.interact()

            # store
            if not self.disable_labels_entirely:
                mail["LLM_label"] = mess_label
            mail["LLM_summary"] = mess_summary
            mail["LLM_short_summary"] = short_mess_summary
            mail["fiat_cost"] = fiat_cost
            mail["tkn_cost"] = tkn_cost
            mail["fiat_cost_summary"] = sum_fiat_cost
            mail["fiat_cost_label"] = label_fiat_cost

            # assign label remotely
            if not self.disable_labels_entirely:
                if not self.dont_labellize:
                    p("Assigning labels.")
                    for lab in mess_label + ["HAL"]:
                        result, _ = self.imap.uid(
                            "STORE", mail["mail_id"], "+X-GM-LABELS", f"({lab})"
                        )
                        assert result == "OK", f"Invalid response: {result}"
                else:
                    p(
                        "Not actually setting the label on the server because 'dont_labellize' is True"
                    )

            # failsafe price check
            total_fiat_cost += fiat_cost
            if total_fiat_cost >= self.total_cost_limit:
                self.interact(
                    f"Total cost so far in {self.fiat_name} is {total_fiat_cost} "
                    f"which is above {self.total_cost_limit}."
                )

    def formating_summary_mail(self):
        "create the html of the summary to send"
        html_mail = """
        <html>
        <style>
             type="text/css">
              li, ol, ul, div, p, a, details, summary {
                font-family: Garamond, sans-serif;
              }
        </style>
        """
        if self.language == "fr":
            html_mail += f"""
                <h1 style="align: center !important; font-family: 'Brush Script MT', cursive;">HAL v{self.VERSION}</h1>
                <h3 style="align: center;">Par Olivier CORNELIS et Aurelien JARDIN</h3>
                <ul>
                    <li style="list-style-type: none;">Nombre de mails: {len(self.inbox_mails)}</li>
                    <li style="list-style-type: none;">LLM utilisé: {self.llm_name}</li>
                    <li style="list-style-type: none;">Coût total: TKN_TOTAL_COST tokens (FIAT_TOTAL_COST {self.fiat_name})</li>
                </ul>
                <ul>
            """
        elif self.language == "en":
            html_mail += f"""
                <h1 style="align: center !important; font-family: 'Brush Script MT', cursive;"><b>HAL v{self.VERSION}</b></h1>
                <h3 style="align: center;">By Olivier CORNELIS et Aurelien JARDIN</h3>
                <ul>
                    <li style="list-style-type: none;">Number of emails: {len(self.inbox_mails)}</li>
                    <li style="list-style-type: none;">LLM used: {self.llm_name}</li>
                    <li style="list-style-type: none;">Total cost: TKN_TOTAL_COST tokens (FIAT_TOTAL_COST {self.fiat_name})</li>
                </ul>
                <ul>
            """
        total_fiat_cost = 0
        total_tkn_cost = 0

        # sort mail by label:
        self.inbox_mails = sorted(
            self.inbox_mails, key=lambda x: x["LLM_label"])
        latest_label = None

        for i_mail, mail in enumerate(
            tqdm(self.inbox_mails, desc="Processing", unit="mail")
        ):
            n_attach = len(mail["attachments"])

            sender = mail["From"]
            # if the sender is something like (Paul <paul@server.org>), parse it
            # explicitely
            # if "<" in sender and ">" in sender:
            #     sender = sender.replace("<", "(").replace(">", ")")
            sender = sender.replace("'", "").replace('"', "")

            if not self.disable_labels_entirely:
                if mail["LLM_label"] != latest_label:
                    html_mail += f"""
                    <li style="list-style-type: none;">
                        Label: <b>{mail['LLM_label']}</b>
                    </li>
                    <li style="list-style-type: none;">&nbsp;</li>
                    """
                    latest_label = mail["LLM_label"]

            if self.language == "fr":
                html_mail += f"""
                <br><br>
                <li style="list-style-type: none;">
                    {i_mail+1}. <b>"{mail['Subject'].title().strip()}"</b> de <b>{sender}</b> le {mail['parsed_date']}
                </li>
                <ul>
                <table bgcolor="#D3D3D3" width="100%" border="0" cellspacing="0" cellpadding="10" style="border-radius: 10px;"><tr><td>
                """
            elif self.language == "en":
                html_mail += f"""
                <br><br>
                <li style="list-style-type: none;">
                    {i_mail+1}. <b>"{mail['Subject'].title().strip()}"</b> from <b>{sender}</b> the {mail['parsed_date']}
                </li>
                <ul>
                <table bgcolor="#D3D3D3" width="100%" border="0" cellspacing="0" cellpadding="10" style="border-radius: 10px;"><tr><td>
                """

            if n_attach:
                if self.language == "fr":
                    html_mail += """
                    <li style="list-style-type: none;">
                        Fichiers joints:
                        <ul>"""
                elif self.language == "en":
                    html_mail += """
                    <li style="list-style-type: none;">
                        Attachments:
                        <ul>"""
                for at in mail["attachments_list"]:
                    html_mail += f"""
                            <li>{at}</li>
                    """
                html_mail += """
                        </ul>
                    </li>"""

            html_mail += f"""
            <li style="list-style-type: none;">
                <ul>
                <li style="list-style-type: none;">
                    {mail['LLM_short_summary']}
                </li>
                <li style="list-style-type: none;">
                    <table bgcolor="#ADD8E6" width="100%" border="0" cellspacing="0" cellpadding="10" style="border-radius: 10px;"><tr><td>
                          {mail['LLM_summary']}
                    </td></tr></table>
                </li>
                    <br>
                    """
            if not self.detailed_price:
                html_mail += """
                <!---
                """
            html_mail += f"""
                <li style="list-style-type: none;">Token cost: {mail['tkn_cost']}</li>
                <li style="list-style-type: none;">Fiat cost for summary: {round(float(mail['fiat_cost_summary']), 5)}</li>
                <li style="list-style-type: none;">Fiat cost for labels: {round(float(mail['fiat_cost_label']), 5)}</li>
                <li style="list-style-type: none;">Fiat cost in {self.fiat_name}: {round(float(mail['fiat_cost']), 5)}</li>
                """
            if not self.detailed_price:
                html_mail += """
                -->
                """
            html_mail += """
                </ul>
            </li>
            </li>
            </td></tr></table>
            </ul>
            <br>
            """
            total_tkn_cost += mail["tkn_cost"]
            total_fiat_cost += mail["fiat_cost"]

        html_mail += """
            </ul>
        </html>
        """
        html_mail = html_mail.replace("TKN_TOTAL_COST", str(total_tkn_cost))
        html_mail = html_mail.replace(
            "FIAT_TOTAL_COST", str(round(total_fiat_cost, 2)))

        html_mail = "".join([item.strip() for item in html_mail.splitlines()])

        self.html_mail = html_mail

    def _mailinfo(self, mailid: str):
        "given a mailid, fetch all data and metadata related to the mail"
        # read the mail
        # state, data = self.imap.uid('fetch', mailid, '(RFC822)')
        # read but without marking as read
        state, data = self.imap.uid("fetch", mailid, "(BODY.PEEK[])")

        assert state == "OK", f"Invalid state: {state}, {data}"
        assert len(data[0]) in [1, 2], f"Unexpected data[0]: {data[0]}"
        raw = email.message_from_bytes(data[0][1])

        metadata = {k: v for k, v in raw.items()}

        metadata["mail_id"] = mailid

        # sometimes the subject etc is encoded
        for col in [
            "Subject",
            "To",
            "Reply-To",
            "Delivered-To",
            "Received",
            "From",
            "Cc",
        ]:
            if col not in metadata:
                continue
            temp = email.header.decode_header(metadata[col])[0][0]
            metadata[col] = decode_item(temp)

        content = []
        if raw.is_multipart():
            metadata["is_multipart"] = True
            for part in raw.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" or content_type == "text/html":
                    temp = part.get_payload(decode=True)
                    content.append(decode_item(temp))
        else:
            metadata["is_multipart"] = False
            temp = raw.get_payload(decode=True)
            content.append(decode_item(temp))

        # get attachments
        attachment_names = []
        for part in raw.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition") is None:
                continue
            filename = part.get_filename()
            if filename:
                temp = email.header.decode_header(filename)[0][0]
                attachment_names.append(decode_item(temp))
        metadata["attachments"] = attachment_names

        metadata["decoded_content"] = content
        return metadata

    def _labelizer(self, content: str):
        "one API call to get the labels to assign to a given mail"
        assert (
            not self.disable_labels_entirely
        ), "Labellizer called even though disable_labels_entirely is True"
        if self.use_cache:
            caller = self.mem.cache(llm_call, ignore=["verbose"])
        else:
            caller = llm_call

        messages = [
            {
                "role": "system",
                "content": dedent(self.labellizer_prompt).strip(),
            },
            {
                "role": "user",
                "content": dedent(content).strip(),
            },
        ]
        answer = caller(
            modelname=self.llm_name,
            messages=messages,
            temperature=0,
            verbose=self.verbose,
        )
        return answer

    def _summarizer(self, mail_content: str, prompt: str):
        "one API call to get the summary of a given mail"
        if self.use_cache:
            caller = self.mem.cache(llm_call, ignore=["verbose"])
        else:
            caller = llm_call

        messages = [
            {
                "role": "system",
                "content": dedent(prompt).strip(),
            },
            {
                "role": "user",
                "content": dedent(mail_content).strip(),
            },
        ]
        answer = caller(
            modelname=self.llm_name,
            messages=messages,
            temperature=0,
            verbose=self.verbose,
        )
        return answer

    def send_summary(self):
        "send the summary"
        p("Mail to send:")
        # p(BeautifulSoup(self.html_mail, "html.parser").get_text())
        p(self.html_mail)
        self.interact()

        if self.dont_send_summary:
            p("Not actually sending the mail because 'dont_send_summary' is True")
            return

        msg = MIMEMultipart()
        msg["From"] = self.inbox_mail
        msg["To"] = ",".join(self.summary_recipients)
        if self.language == "fr":
            msg["Subject"] = f"HAL - Résumé du {timestamp_fr}"
        elif self.language == "en":
            msg["Subject"] = f"HAL - Summary of {timestamp_en}"
        msg.attach(MIMEText(self.html_mail, "html"))
        # assigning labels is not working apparently
        # if "gmail" in self.smtp_domain and "gmail" in self.imap_domain:
        #     msg["X-GM-LABELS"] = "HAL".encode("ascii", "ignore").decode()
        message = msg.as_string()
        self.smtp.sendmail(self.inbox_mail, self.summary_recipients, message)
        self.smtp.quit()

    def exit(self):
        # otherwise fire will display the help page
        sys.exit(0)


def llm_call(
    modelname: str,
    messages: list[dict],
    temperature: float,
    verbose: bool,
):
    "call to the LLM api"
    if verbose:
        p(json.dumps(messages, indent=4, ensure_ascii=False))
    answer = completion(
        model=modelname,
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return answer


def decode_item(item):
    if not isinstance(item, str):
        enc = chardet.detect(item)["encoding"]
        try:
            item = item.decode(enc)
        except:
            item = item.decode()
    item = item.replace("\xa0", " ").replace("\xc2", " ")
    item = ftfy.fix_text(item)
    return item


if __name__ == "__main__":
    kwargs = fire.Fire(lambda **kwargs: kwargs)
    if "help" in kwargs or "h" in kwargs or "usage" in kwargs:
        print(HAL.__init__.__doc__)
    else:
        HAL(**kwargs)

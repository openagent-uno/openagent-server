"""Legal-silence corpus for the support controller dry run.

Two sides of one rule (company/policies/no-legal-response-policy.md): a legal,
copyright, platform-enforcement, regulator or investor communication gets NO
automated message of any kind - not an answer, not "a colleague is taking this
over" - while an ordinary customer who merely says "policy", "terms", "rights"
or "legal" is served normally.

Every entry is plain data so the dry run can turn it into an OperationalCase and
the unit tests can replay the deterministic part without a model:

- ``expect``: ``silence`` (must end in ``legal_silence``), ``handle`` (must
  NOT), or ``observe`` (policy-ambiguous: recorded, never scored).
- ``signal``: which layer is supposed to catch it - ``strong`` (keyword),
  ``sender`` (legal@ handle or "Legal" display name), ``tag`` (thread already
  tagged ``legal``), ``model`` (only weak cues, the classifier decides),
  ``history`` (the legal wording is in an earlier inbound of the same thread).
  ``handle`` entries use ``cue`` when a weak cue sends them to the classifier
  and ``none`` otherwise.
- ``source``: ``synthetic`` or ``real`` (a past Replio thread, with people's
  names and personal addresses replaced).

17-Sep-2026: the YouTube Legal letter got the human-handoff acknowledgement.
"""
from __future__ import annotations

from typing import Any

YOUTUBE_LEGAL_LETTER = (
    "To Whom It May Concern:\n\n"
    "We recently became aware of your product or service, eSound: MP3 Music Player App "
    "(“Your Client”), which is being offered at "
    "https://play.google.com/store/apps/details?id=com.esound.\n"
    "It appears Your Client:\n\n"
    "Allows downloading of YouTube Content\n\n"
    "Is separating audio from the visuals of YouTube Content, meaning you are modifying the "
    "original YouTube Content where the audio of that Content does not play over the visuals "
    "of the Content and vice versa\n\n"
    "Plays YouTube Content in the background, meaning the audio or visual of the Content is not "
    "displayed in the page, tab, or screen that the user is viewing\n\n"
    "Is accessing, reproducing, distributing, or displaying YouTube Content without using the "
    "embeddable YouTube Player. You may only show YouTube videos through the embeddable YouTube player\n\n"
    "As you know, you are required to comply with the YouTube Terms of Service at all times, "
    "which are posted at www.youtube.com/t/terms. Compliance includes, without limitation, not "
    "accessing, modifying, selling, or otherwise using the Services (including any Content) in "
    "violation of the YouTube Terms of Service.\n\n"
    "In addition, Your Client appears to be in violation of the YouTube API Services Terms of "
    "Service and Developer Policies:\n\n"
    "Policy # :III.A.1\nViolation: API Clients did not state in their own terms of use that, by "
    "using those API Clients, users are agreeing to be bound by the YouTube Terms of Service\n\n"
    "Policy # :III.I.9\nViolation: API Client must not create a feature that allows for background play\n\n"
    "We strive to keep YouTube a safe, responsible community, and to encourage respect for the "
    "rights of millions of YouTube creators. If applicable, you must also delete any Content "
    "(including data) that you may have gathered in violation of our terms, policies, and "
    "applicable laws.\n\n"
    "Please immediately correct and cease offering Your Client that violates our terms and "
    "policies within 7 days from the date of this letter.\n\n"
    "If you are accessing or using the YouTube API Services, please provide all the API Project "
    "IDs for Your Client by responding to this letter.\n\n"
    "Sincerely,\nThe YouTube Legal Team"
)

# The same letter with every keyword the strong list knows removed: what is
# left is exactly what a platform enforcement notice looks like to a regex
# that has never seen one - terms, policies, a notice and a deadline.
YOUTUBE_LETTER_WEAK_ONLY = (
    "To Whom It May Concern:\n\n"
    "We recently became aware of your application, which is being offered on Google Play. "
    "It appears your application plays YouTube content in the background and separates the "
    "audio from the visuals of that content.\n\n"
    "As you know, you are required to comply with the YouTube Terms of Service and the API "
    "Services policies at all times. This notice sets out the provisions your "
    "application does not currently meet.\n\n"
    "Please immediately correct your application within 7 days of receipt of this notice and "
    "provide all project identifiers by responding to this message.\n\n"
    "Sincerely,\nYouTube"
)


LEGAL: tuple[dict[str, Any], ...] = (
    # ---- The incident itself, from every angle -------------------------------------------
    {
        "id": "youtube-legal-letter-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "YouTube Terms of Service Violation",
        "author_name": "YouTube Legal",
        "author_handle": "legal-youtube+0jzhndqu2k9tv0y@google.com",
        "body": YOUTUBE_LEGAL_LETTER,
    },
    {
        "id": "youtube-letter-body-only-no-sender", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "web_form", "lang": "en",
        "body": YOUTUBE_LEGAL_LETTER,
    },
    {
        "id": "youtube-letter-weak-cues-only-generic-sender", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Your application", "author_handle": "notices.team@gmail.com",
        "body": YOUTUBE_LETTER_WEAK_ONLY,
    },
    {
        "id": "legal-sender-short-followup-no-keywords", "expect": "silence", "signal": "sender",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Re: Your application", "author_name": "",
        "author_handle": "legal-youtube+0jzhndqu2k9tv0y@google.com",
        "body": "Thanks, we await your confirmation.",
    },
    {
        "id": "display-name-only-legal-sender", "expect": "silence", "signal": "sender",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Your app", "author_name": "Sony Music Legal",
        "author_handle": "notices@sonymusic-mail.example",
        "body": "Hello, please confirm you received our previous message and reply at your earliest convenience.",
    },
    {
        "id": "tagged-legal-thread-innocuous-message", "expect": "silence", "signal": "tag",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Re: follow up", "author_handle": "jordan.m1987@gmail.com",
        "tags": ["legal"],
        "body": "Hi, just checking you got my last email. Have a nice day!",
    },
    {
        "id": "rights-holder-followup-history-only", "expect": "silence", "signal": "history",
        "source": "real", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "WCM x Lyra Music", "author_name": "Publisher licensing manager",
        "author_handle": "licensing.manager@publisher-music.example",
        "history": [
            {"direction": "inbound", "body_text": (
                "Hi,\n\nI'm reaching out on behalf of Warner Chappell Music, the publishing division "
                "of Warner Music Group. I understand that WCM works are available on the Lyra "
                "service so wanted to connect to better understand your current licensing "
                "arrangements and explore how we might work together to ensure all WCM rights "
                "are appropriately covered.\n\nAs a first step, it would be helpful to understand "
                "consumption across the service and annual revenues.")},
        ],
        "body": "Just following up on this, it would be helpful if you could come back to me so we can discuss next steps",
    },
    {
        "id": "rights-holder-followup-tagged", "expect": "silence", "signal": "tag",
        "source": "real", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "WCM x Lyra Music", "author_handle": "licensing.manager@publisher-music.example",
        "tags": ["business", "copyright", "legal", "needs-human"],
        "body": "Just following up on this, it would be helpful if you could come back to me so we can discuss next steps",
    },
    # ---- Platforms ---------------------------------------------------------------------------
    {
        "id": "google-play-unauthorized-youtube-use", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Action Required: Your app is not compliant with Google Play Policies (eSound)",
        "author_name": "Google Play Console", "author_handle": "no-reply-googleplay-developer@google.com",
        "body": (
            "Hi Developers at Spicy Sparks,\n\nAfter review, your app eSound has been found in "
            "violation of the Intellectual Property policy: Unauthorized use of YouTube content. "
            "Apps that facilitate background playback or downloading of YouTube videos are not "
            "allowed on Google Play. Your app will be removed if the issue is not addressed "
            "within 7 days. Please review the Policy Center and submit an update."),
    },
    {
        "id": "apple-app-review-ip-notice", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Your App Store app, Lyra", "author_name": "App Store Notices",
        "author_handle": "appstorenotices@apple.com",
        "body": (
            "Dear Developer,\n\nWe have received a notice from a third party that believes your "
            "app infringes their intellectual property rights. Please contact the complainant "
            "directly and provide Apple with written assurance that your application does not "
            "infringe their rights, or that you are taking steps to resolve the matter, within "
            "14 days. Failure to respond may result in the removal of your app from the App Store."),
    },
    {
        "id": "apple-legal-display-name-routine-wording", "expect": "silence", "signal": "sender",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Case 102938475", "author_name": "Apple Legal",
        "author_handle": "no_reply@email.apple.com",
        "body": "Please find our message in App Store Connect and respond there.",
    },
    # ---- Rights holders, agents, labels ------------------------------------------------------
    {
        "id": "dmca-agent-notice-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "[Notice ID: 878655620] Copyright Infringement Notice under DMCA Act [Application Name: eSound: MP3 Music Player App]",
        "author_name": "Markscan_DMCA", "author_handle": "internet_enforcement@markscan.in",
        "body": (
            "Dear Madam/Sir,\n\nWe, MarkScan, have been duly authorized by Zee Music Company "
            "(Referred as \"Our Client\") to issue this notice to you under the provisions of the "
            "Online Copyright Infringement Liability Limitation Act (OCILLA), a part of the "
            "Digital Millennium Copyright Act 1998 (DMCA).\n\nOur Client is the owner of the "
            "following copyright work(s): Zaalima\n\nWe have found material on your Application "
            "which infringes the copyright of our client. In order to comply with the shared DMCA "
            "notice, you are requested to remove all catalogue owned by Zee Music from your "
            "application or acquire a licence.\n\nWe hereby request you to immediately remove "
            "and/or disable access to the infringing material(s) from your Application.\n\n"
            "Regards,\nMarkScan Internet Enforcement Team"),
    },
    {
        "id": "dmca-agent-escalation-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Re: [Notice ID: 878655620] Copyright Infringement Notice",
        "author_handle": "internet_enforcement@markscan.in",
        "body": (
            "Hi eSound Team,\n\nThank you for your response. Please note that our client holds the "
            "exclusive rights to the reported content. We hereby request that you immediately take "
            "down the content. If it is not removed from your application within the next 24 "
            "hours, we will escalate this matter to Google for permanent suspension of your "
            "application.\n\nRegards"),
    },
    {
        "id": "major-label-takedown-umg", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Unauthorized exploitation of sound recordings",
        "author_handle": "content.protection@umusic.example",
        "body": (
            "Universal Music Group has identified sound recordings owned or exclusively licensed "
            "by UMG made available through your application, including \"Blinding Lights\". "
            "Please disable access to these recordings and confirm in writing."),
    },
    {
        "id": "sony-label-rights-es", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "es",
        "subject": "Reclamación de derechos - Sony Music Entertainment España",
        "author_handle": "antipirateria@sonymusic.example",
        "body": (
            "Estimados señores:\n\nEn nombre de Sony Music Entertainment España, titular de los "
            "derechos de las grabaciones que figuran en el anexo, les requerimos que retiren de "
            "su aplicación el acceso a dichas grabaciones en un plazo de diez días. De lo "
            "contrario ejerceremos las acciones que correspondan.\n\nAtentamente,\nDepartamento Jurídico"),
    },
    {
        # The classifier called this "an ordinary user" in 2 of 2 runs, so it
        # is now caught by the "did not authorise my tracks" phrase.
        "id": "rights-holder-plain-first-message-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Removal of my tracks from Lyra", "author_handle": "artist.office@indie-label.example",
        "body": (
            "I did not authorise my tracks to be on Lyra.\nRemove them,\nunless you start paying I "
            "will not be offering my tracks on Lyra.\nHere's a screenshot of my DistroKid "
            "dashboard, I want all the songs offered on Lyra to be removed.\n"
            "Artist ; Musical Milkshake\nTracks ; All available ones"),
    },
    {
        "id": "rights-holder-seven-days-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Removal of my tracks from Lyra", "author_handle": "artist.office@indie-label.example",
        "body": (
            "No. I will not be providing my channel ID or any further information.\nYour app "
            "scrapes content from YouTube without proper licensing and strips ad revenue from "
            "artists. This violates YouTube's API Developer Policies and constitutes copyright "
            "infringement.\nRemove all tracks by Musical Milkshake immediately or I will escalate "
            "this through Apple App Store review, Google Play policy enforcement, YouTube copyright "
            "complaints, and legal channels.\nYou have 7 days."),
    },
    {
        "id": "artist-instagram-dm-using-my-music", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "lyra", "channel": "instagram_dm", "lang": "en",
        "author_handle": "instagram:17841400000000001",
        "body": "hey, you're using my music on your app without asking me. take it down or pay me",
    },
    {
        "id": "label-manager-discord", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "lyra", "channel": "discord", "lang": "en",
        "author_handle": "discord:100000000000000001",
        "body": "I manage a record label and our catalogue is on Lyra with no deal. Who do I talk to about royalties?",
    },
    {
        "id": "rights-holder-playstore-review", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "playstore_reviews", "lang": "en",
        "body": (
            "This app streams my copyrighted songs without any license. I am the artist. Remove my "
            "music now.\n\n---\napp_version: 5.2.6\nos: Android 14\nreviewer_language: en"),
    },
    {
        "id": "on-behalf-of-our-client-no-lawyer-word", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Northbridge Records Ltd / eSound", "author_handle": "office@hartwell-partners.example",
        "body": (
            "Dear Sirs,\n\nWe write on behalf of our client, Northbridge Records Ltd, the exclusive "
            "licensee of the works listed in the schedule. Our client has not consented to the "
            "availability of those works through your service. We demand that you confirm, within "
            "10 days, the steps you will take to stop making them available, failing which our "
            "client reserves all of its rights.\n\nYours faithfully"),
    },
    {
        "id": "cease-and-desist-en", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Cease and Desist", "author_handle": "j.doyle@doyle-law.example",
        "body": "This letter constitutes formal demand that you cease and desist from using our client's name in your app store listing.",
    },
    {
        "id": "trademark-brand-protection-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Infringement of Spotify's Intellectual Property Rights / esound.app",
        "author_handle": "enf.spotify.2540811@enftracer.ai",
        "body": (
            "Please find the necessary information below for a takedown notice against an "
            "infringing material offered herein.\n\nTrademark Rights: Spotify Wordmark\n\n"
            "Infringing URLs: https://esound.app/\n\nThe material uses Spotify's well-known "
            "trademarks without authorization. Please remove these images of the Spotify UI.\n\n"
            "Tracer Brand Protection"),
    },
    # ---- Collecting societies and Italian authorities ---------------------------------------
    {
        "id": "fpm-diffida-it", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "it",
        "subject": "Diffida - applicazione eSound", "author_handle": "segreteria@fpm-antipirateria.example",
        "body": (
            "Spett.le Spicy Sparks,\n\nla FPM - Federazione contro la Pirateria Musicale e "
            "Multimediale, per conto delle case discografiche associate, rileva che l'applicazione "
            "eSound consente l'ascolto di fonogrammi in violazione dei diritti dei produttori. "
            "Con la presente Vi diffidiamo dal proseguire tale attività e Vi invitiamo a fornire "
            "riscontro entro 15 giorni dal ricevimento.\n\nDistinti saluti"),
    },
    {
        "id": "siae-licenza-it", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "it",
        "subject": "Utilizzo del repertorio tutelato", "author_handle": "multimedialita@siae.example",
        "body": (
            "Gentili signori,\n\nSIAE ha rilevato che il servizio eSound mette a disposizione del "
            "pubblico opere del repertorio amministrato senza la relativa licenza. Vi invitiamo a "
            "contattare l'ufficio Multimedialità per regolarizzare la posizione entro 30 giorni."),
    },
    {
        "id": "agcom-procedimento-weak-it", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "it",
        "subject": "Comunicazione di avvio del procedimento DDA/123/26",
        "author_handle": "protocollo@agcom-pec.example",
        "body": (
            "Oggetto: comunicazione di avvio del procedimento ai sensi del Regolamento in materia "
            "di tutela del diritto d'autore sulle reti di comunicazione elettronica.\n\nA seguito "
            "dell'istanza presentata da un soggetto legittimato, la Direzione competente comunica "
            "l'avvio del procedimento nei confronti del servizio eSound. Eventuali controdeduzioni "
            "potranno essere trasmesse entro 5 giorni dalla ricezione della presente, secondo la "
            "normativa vigente.\n\nIl Direttore"),
    },
    {
        "id": "garante-privacy-richiesta-it", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "it",
        "subject": "Richiesta di informazioni - fascicolo n. 45678",
        "author_handle": "protocollo@gpdp-pec.example",
        "body": (
            "Il Garante per la protezione dei dati personali, in relazione a una segnalazione "
            "pervenuta da un interessato, chiede a codesta società di fornire, ai sensi dell'art. "
            "157 del Codice, informazioni sul trattamento dei dati degli utenti dell'app eSound e "
            "sul riscontro dato alla richiesta dell'interessato. Si invita a fornire riscontro "
            "entro 20 giorni dalla ricezione della presente."),
    },
    {
        "id": "consumer-threatens-lawyer-real-it", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "it",
        "subject": "Ennesimo Reclamo", "author_handle": "cliente.reclamo@virgilio.example",
        "body": (
            "Salve vi scrivo per la 20 volta, la situazione sta continuando anche questi mesi due "
            "rinnovi cioè due addebiti, la cosa va avanti da mesi ormai, risolvete la cosa e fate un "
            "rimborso o vi faccio scrivere dall'avvocato"),
    },
    {
        "id": "studio-legale-web-form-it", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "web_form", "lang": "it",
        "subject": "Other request",
        "author_handle": "segreteria@studiolegalerossi.example",
        "body": (
            "Scrivo per conto dello Studio Legale Rossi in rappresentanza di un'artista i cui brani "
            "sono disponibili nella vostra app. Vi chiediamo di indicarci un referente.\n---\n"
            "app_version: 5.2.6\nos: Android 14"),
    },
    {
        "id": "gema-abmahnung-de", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "de",
        "subject": "Abmahnung wegen Urheberrechtsverletzung", "author_handle": "lizenzierung@gema.example",
        "body": (
            "Sehr geehrte Damen und Herren,\n\ndie GEMA hat festgestellt, dass Ihre Anwendung "
            "Musikwerke aus unserem Repertoire ohne Lizenz öffentlich zugänglich macht. Wir fordern "
            "Sie auf, die Nutzung einzustellen und binnen 14 Tagen eine strafbewehrte "
            "Unterlassungserklärung abzugeben.\n\nMit freundlichen Grüßen"),
    },
    {
        "id": "kanzlei-mandantin-weak-de", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "de",
        "subject": "Unsere Mandantin ./. Lyra", "author_handle": "sekretariat@kanzlei-weber.example",
        "body": (
            "Sehr geehrte Damen und Herren,\n\nwir zeigen an, dass wir die rechtlichen Interessen "
            "unserer Mandantin vertreten. Ihre App stellt Aufnahmen unserer Mandantin ohne deren "
            "Zustimmung bereit. Wir fordern Sie auf, dies bis zum Ablauf von 10 Tagen abzustellen "
            "und uns eine Unterlassungserklärung zukommen zu lassen.\n\nRechtsanwalt Dr. Weber"),
    },
    {
        "id": "sacem-fr", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "fr",
        "subject": "Exploitation d'œuvres du répertoire SACEM", "author_handle": "licences.numeriques@sacem.example",
        "body": (
            "Madame, Monsieur,\n\nLa SACEM a constaté que votre application met à la disposition du "
            "public des œuvres de son répertoire sans autorisation. Nous vous remercions de prendre "
            "contact avec nos services afin de régulariser cette situation sous 30 jours."),
    },
    {
        "id": "avocat-mise-en-demeure-fr", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "fr",
        "subject": "Mise en demeure", "author_handle": "cabinet@dupont-avocats.example",
        "body": (
            "Madame, Monsieur,\n\nJe suis le conseil de la société Label Bleu, titulaire des droits "
            "sur les enregistrements diffusés par votre application. Par la présente, je vous mets "
            "en demeure de cesser toute mise à disposition de ces enregistrements sous huitaine, à "
            "défaut de quoi ma cliente saisira la juridiction compétente.\n\nMaître Dupont, avocat"),
    },
    {
        "id": "notificacao-extrajudicial-pt", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "pt",
        "subject": "Notificação extrajudicial", "author_handle": "contato@silva-advogados.example",
        "body": (
            "Prezados,\n\nNa qualidade de advogados da gravadora Som Brasil Ltda., notificamos "
            "extrajudicialmente V.Sas. para que cessem, no prazo de 10 (dez) dias, a disponibilização "
            "não autorizada de fonogramas de titularidade de nossa cliente no aplicativo eSound, "
            "sob pena de adoção das medidas judiciais cabíveis.\n\nAtenciosamente"),
    },
    {
        "id": "abogado-messenger-es", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "messenger", "lang": "es",
        "author_handle": "messenger:100000000000000002",
        "body": (
            "Buenas, soy el abogado de un cantante cuyas canciones aparecen en su app sin "
            "autorización. Necesito el correo de su representante legal para enviarles un "
            "requerimiento formal."),
    },
    {
        "id": "citacion-juzgado-es", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "es",
        "subject": "Cédula de citación - Procedimiento ordinario 412/2026",
        "author_handle": "notificaciones@juzgado-madrid.example",
        "body": (
            "Por medio de la presente se cita a la entidad titular de la aplicación eSound a "
            "comparecer ante el Juzgado de lo Mercantil n.º 3 de Madrid en el procedimiento "
            "ordinario 412/2026, a instancia de Discos del Sur S.L. Se le advierte que, de no "
            "comparecer, será declarada en rebeldía."),
    },
    {
        "id": "collecting-society-juridico-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "eSound - EMMAC SACM", "author_name": "Legal department",
        "author_handle": "depto.juridico@collecting-society.example",
        "body": (
            "Dear eSound Team,\nOn behalf of EMMACSACM, the Mexican one-stop shop created by the "
            "Sociedad de Autores y Compositores de México (SACM) and Editores Mexicanos de Musica "
            "(EMMAC) to facilitate the granting of licenses for the digital exploitation of musical "
            "works in Mexico, by the provisions of the Mexican Copyright Law.\nWe confirmed that "
            "eSound is using and exploiting in Mexico musical works without the corresponding "
            "license. We must reach an agreement as soon as possible to avoid any inconvenience.\n"
            "Kind regards,\nAbogado | Attorney"),
    },
    {
        "id": "data-protection-authority-en", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "en",
        "subject": "Complaint reference C-2026-0912", "author_handle": "info@dataprotection-commission.example",
        "body": (
            "Dear Sir or Madam,\n\nThe Commission has received a complaint from an individual "
            "concerning the processing of their personal data by the Lyra application. Pursuant to "
            "its powers under Article 58 GDPR, the Commission requests that you provide a full "
            "account of the processing and of your handling of the individual's request within 28 "
            "days."),
    },
    # ---- Investors -----------------------------------------------------------------------------
    {
        "id": "acquisition-inquiry-real", "expect": "silence", "signal": "strong",
        "source": "real", "product": "esound", "channel": "email_imap", "lang": "en",
        "subject": "Acquisition inquiry for eSound", "author_handle": "deals@app-acquirer.example",
        "body": (
            "Hi eSound: MP3 Music Player App,\n\nI came across your app eSound and wanted to reach "
            "out.\n\nWe're currently looking to acquire audio based apps and your app looks like a "
            "strong fit for what we're exploring.\n\nWe focus on improving overall performance post "
            "acquisition, and we're able to move quickly on opportunities like this.\n\nWould you "
            "be open to a short call to explore this?"),
    },
    {
        "id": "vc-fund-it", "expect": "silence", "signal": "strong",
        "source": "synthetic", "product": "lyra", "channel": "instagram_dm", "lang": "it",
        "author_handle": "instagram:17841400000000002",
        "body": "Ciao! Siamo un fondo di venture capital e stiamo valutando un investimento in Lyra, siete aperti a un round?",
    },
    # ---- Non-Latin scripts ---------------------------------------------------------------------
    {
        "id": "copyright-notice-ru", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "esound", "channel": "email_imap", "lang": "ru",
        "subject": "Уведомление о нарушении авторских прав", "author_handle": "pravo@label-ru.example",
        "body": (
            "Уважаемые господа,\n\nнаша компания является правообладателем фонограмм, которые без "
            "разрешения доступны в вашем приложении eSound. Требуем прекратить их использование в "
            "течение 10 дней с момента получения настоящего уведомления, в противном случае мы "
            "обратимся в суд."),
    },
    {
        "id": "copyright-notice-ja", "expect": "silence", "signal": "model",
        "source": "synthetic", "product": "lyra", "channel": "email_imap", "lang": "ja",
        "subject": "著作権侵害に関する通知", "author_handle": "rights@jp-label.example",
        "body": (
            "ご担当者様\n\n弊社が権利を保有する楽曲が、許諾なく貴社アプリ「Lyra」で配信されています。"
            "本通知の受領後14日以内に当該楽曲の配信を停止するよう求めます。対応いただけない場合は法的措置を検討します。"),
    },
)


HANDLE: tuple[dict[str, Any], ...] = (
    {"id": "refund-policy-question", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "body": "What is your refund policy? I paid yesterday for the yearly plan by mistake."},
    {"id": "agree-terms-premium-broken", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "web_form", "lang": "en",
     "body": "I agree to the terms and I paid, but premium doesn't work, I still see ads."},
    {"id": "is-it-legal-in-germany", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "instagram_dm", "lang": "en",
     "body": "is it legal to use eSound in Germany?"},
    {"id": "rights-to-my-playlist", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "discord", "lang": "en",
     "body": "do I have the rights to share my playlist with friends? it says private"},
    {"id": "privacy-policy-link-broken", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "email_imap", "lang": "en",
     "body": "Your privacy policy link on the website is broken, it gives a 404."},
    {"id": "gdpr-deletion-request", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "body": "Under the GDPR I request that you delete my account and all my personal data."},
    # Decision: an ordinary, joking customer. "violates" is a strong word today,
    # so this measures the price of keeping it strong.
    {"id": "violates-my-patience-ads", "expect": "handle", "signal": "none", "source": "synthetic",
     "product": "esound", "channel": "playstore_reviews", "lang": "en",
     "body": "the app violates my patience lol ads everywhere\n\n---\napp_version: 5.2.6\nos: Android 14\nreviewer_language: en"},
    {"id": "notice-downloads-stopped", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "messenger", "lang": "en",
     "body": "notice: my downloads stopped since yesterday"},
    {"id": "termini-abbonamento-non-chiari", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "it",
     "body": "i termini dell'abbonamento non sono chiari, si rinnova da solo?"},
    {"id": "politica-reembolso-es", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "instagram_dm", "lang": "es",
     "body": "no entiendo la política de reembolso, pagué dos veces"},
    {"id": "droits-acces-playlist-fr", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "email_imap", "lang": "fr",
     "body": "Bonjour, je n'ai plus les droits d'accès à ma playlist partagée, pouvez-vous m'aider ?"},
    {"id": "rechtlich-erlaubt-de", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "de",
     "body": "Hallo, ist die App rechtlich in Österreich erlaubt? Ich will kein Problem bekommen."},
    {"id": "politica-privacidade-pt", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "web_form", "lang": "pt",
     "body": "a política de privacidade não abre no app, fica carregando"},
    {"id": "notification-of-new-episodes", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "body": "The notification of new podcast episodes doesn't arrive anymore."},
    {"id": "deadline-party-tonight", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "discord", "lang": "en",
     "body": "deadline is tonight, my party playlist won't load!!"},
    {"id": "on-behalf-of-my-mom", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "body": "I'm writing on behalf of my mom, she can't log in to her account anymore."},
    {"id": "demand-refund-crashes", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "instagram_dm", "lang": "en",
     "body": "I demand a refund for this month, the app keeps crashing"},
    {"id": "terms-page-404-web-form", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "web_form", "lang": "en",
     "body": "Other request: the terms of use page gives 404\n---\napp_version: 5.2.6\nos: iOS 19"},
    {"id": "politica-anuncios-review-es", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "playstore_reviews", "lang": "es",
     "body": "Muy buena app pero la política de anuncios es excesiva\n\n---\napp_version: 5.2.6\nos: Android 13\nreviewer_language: es"},
    {"id": "basketball-court", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "instagram_dm", "lang": "en",
     "body": "when I play basketball on the court the music stops after 5 minutes"},
    {"id": "android-auto-comply", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "body": "The app does not comply with my car display in Android Auto, the buttons are cut."},
    {"id": "diritti-rimborso-doppio-addebito-it", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "esound", "channel": "messenger", "lang": "it",
     "body": "quali sono i miei diritti? mi avete addebitato due volte l'abbonamento"},
    {"id": "legal-name-change-account", "expect": "handle", "signal": "cue", "source": "synthetic",
     "product": "lyra", "channel": "email_imap", "lang": "en",
     "body": "I changed my legal name, how do I update the name on my Lyra profile?"},
    # Real traffic (names removed). All four say "illegal", which is a strong
    # word since 0.21.10: each one measures a customer who would be silenced.
    {"id": "review-looks-like-illegal-website-real", "expect": "handle", "signal": "none", "source": "real",
     "product": "lyra", "channel": "playstore_reviews", "lang": "en",
     "body": "Now there are ads, so I'm not interested anymore... it's really laggy, plus there are ads so it looks like an illegal website\n\n---\napp_version: 1.4.15\nos: Android 13\nreviewer_language: en"},
    {"id": "review-has-to-be-illegal-praise-real", "expect": "handle", "signal": "none", "source": "real",
     "product": "esound", "channel": "playstore_reviews", "lang": "en",
     "body": "IT HAS TO BE ILLEGAL! This app is amazing \U0001F60D The music sounds great \U0001F603 I recommend it for long trips\n\n---\napp_version: 5.2.6\nos: Android 14\nreviewer_language: en"},
    {"id": "charged-not-available-thats-illegal-real", "expect": "handle", "signal": "none", "source": "real",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "subject": "Stop suscripción",
     "body": "You are charging me and your service is no Aveilable in my country , please remove me from your subscription as soon that's illegal"},
    {"id": "is-it-legal-or-pirating-real", "expect": "handle", "signal": "none", "source": "real",
     "product": "esound", "channel": "email_imap", "lang": "en",
     "subject": "is it legal or is it considered pirating to download music from eSound?",
     "body": "I'm just wondering is it legal or is it considered pirating to download music from the standard and also the YouTube part of this eSound music app ? I hope it's not pirating or illegal because I don't want to end up in jail for doing it."},
    {"id": "ethics-question-real", "expect": "handle", "signal": "none", "source": "real",
     "product": "lyra", "channel": "email_imap", "lang": "en", "subject": "Ethics",
     "body": "Hello,\nI wanted to ask if you guys would say you're a more ethical alternative to apps like spotify? Are you against things such as the illegal occupation of Palestine?"},
    # Policy-ambiguous: the hard rule silences anything that "mentions" copyright,
    # yet this is a listener describing a playback error. Recorded, not scored.
    {"id": "song-copyright-issue-review-real", "expect": "observe", "signal": "none", "source": "real",
     "product": "esound", "channel": "playstore_reviews", "lang": "en",
     "body": "It no longer lets me skip to the next song when one finishes. And when there's a song with copyright issues, it gets in the way quite a bit.\n\n---\napp_version: 5.2.6\nos: Android 14\nreviewer_language: en"},
)


ALL: tuple[dict[str, Any], ...] = LEGAL + HANDLE

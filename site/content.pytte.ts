// Copy for mathiaskarlsson.se/pytte in Swedish and English, written in the smallestllm session
// (2026-11-25) from the lab log and the report. Plain data, no imports: adapt it to the site's
// content schema. Every number here is in paper/pytte.pdf; change them only together with it.
// Voice: Mathias, first person, plain and honest. No superlatives the report does not support.

export const pytte = {
  sv: {
    meta: {
      title: "Pytte: en pytteliten språkmodell som skriver svenska",
      description:
        "Ett hobbyprojekt: sex små språkmodeller byggda från noll och tränade på en laptop. Prova dem direkt i webbläsaren, läs rapporten och ladda ner allt.",
    },
    card: {
      // for the grid on /sidoprojekt
      title: "Pytte",
      tag: "Hobbyprojekt · AI",
      body: "En pytteliten språkmodell som skriver svenska sagor. Byggd från noll och tränad på min laptop, för att förstå hur en språkmodell faktiskt fungerar. Prova den i webbläsaren.",
      cta: "Till Pytte",
    },
    hero: {
      eyebrow: "Hobbyprojekt",
      title: "Pytte",
      lead: "En pytteliten språkmodell som skriver svenska sagor. Jag byggde den från noll och tränade den på min laptop, för att förstå hur en språkmodell faktiskt fungerar.",
      note: "Allt nedan körs i din webbläsare. Inget skickas någonstans.",
    },
    demo: {
      tabs: { grow: "Sagan växer", chat: "Prata med Pytte", inside: "Under huven" },
      grow: {
        lead: "Samma början, samma slumpfrö, sex modeller. Dra i reglaget och se språket ta form: från lösa meningar vid 0,3 miljoner parametrar till en saga vid 7.",
        promptLabel: "Börja sagan",
        promptDefault: "Det var en gång en liten älg som",
        run: "Skriv sagan",
        reseed: "Nytt frö",
        sliderLabel: "Modellstorlek",
      },
      chat: {
        lead: "Be om en saga: Berätta en saga om en räv som vill flyga. Fråga sedan: Vad händer sedan? Pytte kan bara en sak, sagor för små barn, och följer namnet du ger oftare än föremålet.",
        placeholder: "Berätta en saga om…",
        send: "Skicka",
        reset: "Börja om",
        who: "Pytte",
      },
      inside: {
        lead: "Modellen ser inte bokstäver utan token: bitar av ord som den lärt sig från sagorna. Skriv något och se hur det delas upp, och vilka token modellen tror kommer sedan.",
        textLabel: "Text",
        textDefault: "Midsommarafton åt Saga en",
        tokens: "Token",
        next: "Nästa token",
      },
      status: {
        loading: "laddar…",
        params: "parametrar",
        tokens: "token",
        perSecond: "token/s",
        seed: "frö",
        error: "Fel",
      },
      footnote: "Modellfilerna är 0,4–16 MB och laddas först när du väljer dem.",
    },
    why: {
      title: "Varför jag gjorde det här",
      body: [
        "Jag ville förstå varje del av en språkmodell: hur text blir token, vad attention gör, vad ett lager är och varför modellen till slut kan skriva en mening. Det enda sättet jag vet att förstå något på riktigt är att bygga det.",
        "Så jag byggde en. Inga färdiga modeller, inga köpta tjänster, ingen budget alls. Samma sorts arkitektur som de stora modellerna, bara tusentals gånger mindre, tränad på min MacBook på mellan fem minuter och en halvtimme.",
        "Allt dokumenterades medan det byggdes: varje körning, varje mätning och varje misstag står i en logg. Så lär jag mig hela tiden, och det är lärdomarna jag vill dela.",
      ],
    },
    facts: {
      title: "Pytte i siffror",
      items: [
        { value: "6", label: "modeller, från 0,3 till 15,7 miljoner parametrar" },
        { value: "120 000", label: "svenska barnsagor som träningsdata, genererade lokalt" },
        { value: "5–30 min", label: "träningstid per modell på en MacBook Pro" },
        { value: "0,4–16 MB", label: "per modell, i en enda fil" },
        { value: "0 kr", label: "i budget. Bara ström." },
      ],
    },
    learned: {
      title: "Vad jag lärde mig",
      items: [
        {
          title: "Det går med förvånansvärt lite",
          body: "En modell med en tredjedels miljon parametrar skriver grammatiska svenska meningar. Vid ungefär en miljon blir det sammanhängande små sagor. Den modellen är 1,4 MB och tränas på åtta minuter.",
        },
        {
          title: "Datan avgör mer än modellen",
          body: "Mina första modeller kunde inte böja i plural, oavsett storlek. Sagorna handlade om ett barn och en boll, så de hade knappt sett en plural. När 40 procent av de nya sagorna skrevs med flera saker i plural klarade alla modeller det, även den minsta. Det modellen kan är det datan visar.",
        },
        {
          title: "En modell kan lära sig en mall i stället för ett språk",
          body: "Den första versionen av datan var för enformig: var sjätte saga började med samma mening. Modellerna såg lysande ut i mina mätningar men föll ihop på friare text. Jag upptäckte det först när jag gjorde en andra, mer varierad omgång sagor.",
        },
        {
          title: "Mer variation kostar sammanhang",
          body: "Sagor med två eller tre barn gav bättre grammatik men modellerna började blanda ihop vem som var vem. En blandning där de nya sagorna bara har ett barn behöll det mesta av båda.",
        },
        {
          title: "Siffror ljuger om man inte räknar två gånger",
          body: "Alla fel jag hittade hittades för att två tal inte gick ihop: ett mått delat med fel antal, en klocka som räknade medan datorn sov, skräptecken i tolv procent av träningsdatan. Felen och rättningarna står i rapporten.",
        },
      ],
    },
    unknown: {
      title: "Vad jag inte vet",
      lead: "Det här är ett hobbyprojekt, inte forskning. Jag vill vara tydlig med var gränserna går.",
      items: [
        "Jag vet inte hur bra Pytte är jämfört med andras små modeller. Jag har bara mätt mot mina egna.",
        "Ingen människa har betygsatt sagorna systematiskt. Betygen kommer från en annan AI-modell, som dessutom är släkt med den som skrev träningsdatan.",
        "Nästan allt är tränat en enda gång. Små skillnader mellan modellerna kan vara slump.",
        "Pytte kan bara barnsagor med några tusen ord. Den vet ingenting om världen och går inte att använda till något nyttigt.",
        "Ingen utomstående har granskat koden, datan eller slutsatserna. Hittar du fel vill jag gärna veta.",
        "Koden och rapporten skrevs tillsammans med en AI-assistent, Claude Code. Jag bestämde vad som skulle göras, läste sagorna och godkände varje steg. Rapporten redovisar vem som gjorde vad.",
      ],
    },
    downloads: {
      title: "Ladda ner",
      report: {
        title: "Rapporten",
        body: "19 sidor på engelska med svensk sammanfattning: data, modeller, mätningar, tjugo fynd, begränsningar och misstag.",
        cta: "Ladda ner rapporten (PDF)",
      },
      models: {
        title: "Modellerna",
        body: "Varje fil innehåller modell och ordlista och är allt som behövs för att köra den. Chattversionerna tar en begäran, basversionerna fortsätter en text.",
        columns: { model: "Modell", params: "Parametrar", size: "Storlek", base: "Bas", chat: "Chatt" },
        download: "Ladda ner",
      },
      run: {
        title: "Kör den på din egen dator",
        body: "Ett enda C-program utan beroenden. Fungerar på Mac och Linux.",
      },
      code: {
        title: "Koden och loggen",
        body: "All kod, receptet för träningsdatan, mätningarna och loggen över hela arbetet finns på GitHub.",
        cta: "Till GitHub",
      },
      data: {
        title: "Träningsdatan",
        body: "SagoSvenska: 120 000 svenska barnsagor med receptet som skapade varje saga. Fri att använda med angivande av källa (CC BY 4.0).",
        cta: "Ladda ner datan (zip, 49 MB)",
      },
      license: "Kod och modeller: Apache-2.0. Data: CC BY 4.0.",
    },
    back: "Till sidoprojekten",
  },

  en: {
    meta: {
      title: "Pytte: a tiny language model that writes Swedish",
      description:
        "A hobby project: six small language models built from scratch and trained on a laptop. Try them in your browser, read the report and download everything.",
    },
    card: {
      title: "Pytte",
      tag: "Hobby project · AI",
      body: "A tiny language model that writes Swedish children's stories. Built from scratch and trained on my laptop, to understand how a language model actually works. Try it in your browser.",
      cta: "See Pytte",
    },
    hero: {
      eyebrow: "Hobby project",
      title: "Pytte",
      lead: "A tiny language model that writes Swedish children's stories. I built it from scratch and trained it on my laptop, to understand how a language model actually works.",
      note: "Everything below runs in your browser. Nothing is sent anywhere.",
    },
    demo: {
      tabs: { grow: "The story grows", chat: "Talk to Pytte", inside: "Under the hood" },
      grow: {
        lead: "The same opening, the same random seed, six models. Drag the slider and watch language take shape: from loose sentences at 0.3 million parameters to a story at 7. Pytte writes in Swedish; “Det var en gång” means “Once upon a time”.",
        promptLabel: "Start the story (in Swedish)",
        promptDefault: "Det var en gång en liten älg som",
        run: "Write the story",
        reseed: "New seed",
        sliderLabel: "Model size",
      },
      chat: {
        lead: "Ask for a story in Swedish: Berätta en saga om en räv som vill flyga (Tell a story about a fox who wants to fly). Then ask: Vad händer sedan? (What happens next?) Pytte can do one thing, stories for small children, and follows the name you give more often than the object.",
        placeholder: "Berätta en saga om…",
        send: "Send",
        reset: "Start over",
        who: "Pytte",
      },
      inside: {
        lead: "The model does not see letters but tokens: pieces of words it learned from the stories. Type something in Swedish and see how it is split, and which tokens the model expects next.",
        textLabel: "Text",
        textDefault: "Midsommarafton åt Saga en",
        tokens: "Tokens",
        next: "Next token",
      },
      status: {
        loading: "loading…",
        params: "parameters",
        tokens: "tokens",
        perSecond: "tokens/s",
        seed: "seed",
        error: "Error",
      },
      footnote: "The model files are 0.4–16 MB and load only when you pick them.",
    },
    why: {
      title: "Why I did this",
      body: [
        "I wanted to understand every part of a language model: how text becomes tokens, what attention does, what a layer is, and why the model can end up writing a sentence. The only way I know to really understand something is to build it.",
        "So I built one. No existing models, no paid services, no budget at all. The same kind of architecture as the large models, only thousands of times smaller, trained on my MacBook in between five minutes and half an hour.",
        "Everything was documented while it was built: every run, every measurement and every mistake is in a log. That way I keep learning the whole time, and the lessons are what I want to share.",
      ],
    },
    facts: {
      title: "Pytte in numbers",
      items: [
        { value: "6", label: "models, from 0.3 to 15.7 million parameters" },
        { value: "120,000", label: "Swedish children's stories as training data, generated locally" },
        { value: "5–30 min", label: "training time per model on a MacBook Pro" },
        { value: "0.4–16 MB", label: "per model, in a single file" },
        { value: "0 kr", label: "budget. Only electricity." },
      ],
    },
    learned: {
      title: "What I learned",
      items: [
        {
          title: "It takes surprisingly little",
          body: "A model with a third of a million parameters writes grammatical Swedish sentences. At about one million it writes short, coherent stories. That model is 1.4 MB and trains in eight minutes.",
        },
        {
          title: "The data decides more than the model",
          body: "My first models could not form plurals, whatever their size. The stories were about one child and one ball, so they had hardly seen a plural. When 40 percent of the new stories were written with several things in the plural, every model got it right, even the smallest. A model can do what its data shows.",
        },
        {
          title: "A model can learn a template instead of a language",
          body: "The first version of the data was too uniform: every sixth story opened with the same sentence. The models looked excellent in my measurements but fell apart on freer text. I only found out when I made a second, more varied batch of stories.",
        },
        {
          title: "Variety costs coherence",
          body: "Stories with two or three children gave better grammar, but the models started mixing up who was who. A mix in which the new stories have a single child kept most of both.",
        },
        {
          title: "Numbers lie unless you count twice",
          body: "Every error I found was found because two numbers did not agree: a metric divided by the wrong count, a clock that kept running while the laptop slept, junk characters in twelve percent of the training data. The errors and the fixes are in the report.",
        },
      ],
    },
    unknown: {
      title: "What I do not know",
      lead: "This is a hobby project, not research. I want to be clear about where the limits are.",
      items: [
        "I do not know how good Pytte is compared with other people's small models. I have only measured against my own.",
        "No person has rated the stories systematically. The grades come from another AI model, which is also a relative of the one that wrote the training data.",
        "Almost everything was trained once. Small differences between the models may be chance.",
        "Pytte knows only children's stories with a few thousand words. It knows nothing about the world and cannot be used for anything practical.",
        "Nobody outside the project has reviewed the code, the data or the conclusions. If you find a mistake, I would like to know.",
        "The code and the report were written together with an AI assistant, Claude Code. I decided what to do, read the stories and approved each step. The report says who did what.",
      ],
    },
    downloads: {
      title: "Download",
      report: {
        title: "The report",
        body: "19 pages in English with a Swedish summary: data, models, measurements, twenty findings, limitations and mistakes.",
        cta: "Download the report (PDF)",
      },
      models: {
        title: "The models",
        body: "Each file contains the model and its vocabulary and is all you need to run it. The chat versions take a request; the base versions continue a text.",
        columns: { model: "Model", params: "Parameters", size: "Size", base: "Base", chat: "Chat" },
        download: "Download",
      },
      run: {
        title: "Run it on your own computer",
        body: "One C program with no dependencies. Works on Mac and Linux.",
      },
      code: {
        title: "The code and the log",
        body: "All the code, the recipe for the training data, the measurements and the log of the whole project are on GitHub.",
        cta: "Go to GitHub",
      },
      data: {
        title: "The training data",
        body: "SagoSvenska: 120,000 Swedish children's stories, each with the recipe that produced it. Free to use with attribution (CC BY 4.0).",
        cta: "Download the data (zip, 49 MB)",
      },
      license: "Code and models: Apache-2.0. Data: CC BY 4.0.",
    },
    back: "Back to side projects",
  },

  // Language-independent data for the downloads table and the size picker.
  models: [
    { key: "0.3m", label: { sv: "0,3 M", en: "0.3M" }, params: { sv: "0,33 miljoner", en: "0.33 million" }, mb: 0.38 },
    { key: "1m", label: { sv: "1 M", en: "1M" }, params: { sv: "1,33 miljoner", en: "1.33 million" }, mb: 1.44 },
    { key: "2m", label: { sv: "2 M", en: "2M" }, params: { sv: "2,24 miljoner", en: "2.24 million" }, mb: 2.37 },
    { key: "3m", label: { sv: "3 M", en: "3M" }, params: { sv: "3,44 miljoner", en: "3.44 million" }, mb: 3.59 },
    { key: "7m", label: { sv: "7 M", en: "7M" }, params: { sv: "7,16 miljoner", en: "7.16 million" }, mb: 7.33 },
    { key: "15m", label: { sv: "15 M", en: "15M" }, params: { sv: "15,74 miljoner", en: "15.74 million" }, mb: 15.97 },
  ],
  // file names: `pytte-${key}.bin` (base) and `pytte-${key}-chat.bin` (chat)
  runCommands: ["cc -O3 -o run run.c -lm", "./run pytte-7m-chat.bin -chat"],
  github: "https://github.com/dragon6sic6/pytte", // public since 2026-11-25
  dataRelease: "https://github.com/dragon6sic6/pytte/releases/tag/sagosvenska-1.1", // SagoSvenska 1.1, published 2026-11-25
  dataZip: "https://github.com/dragon6sic6/pytte/releases/download/sagosvenska-1.1/sagosvenska.zip",
  reportFile: "pytte.pdf",
};

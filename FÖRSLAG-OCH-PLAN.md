# Pytte: förslag och plan

*Sammanställt 2026-11-06 utifrån de tre researchrapporterna i mappen, kompletterat med egen verifiering mot Hugging Face, PyPI och webben samt en inventering av din Mac.*

*Avsnitt 1–9 är planen som Claude skrev på mitt uppdrag den 6 november, innan arbetet började, lätt redigerad: delen om lansering och spridning är borttagen. Mycket av planen blev inte av; avsnitt 23 berättar varför. Från avsnitt 10 är det arbetsloggen.*

---

## 1. Kortversionen

Gör projektet. Men gör det som **"den minsta språkmodellen som skriver sammanhängande svenska, tränad från grunden på en MacBook"**, inte som "världens minsta LLM".

Tre skäl:

1. **"Världens minsta" går inte att försvara.** Det finns redan transformers med 25 000 parametrar på Commodore 64 och 62 000 på Game Boy. Ett "minst"-påstående utan kvalitetsgolv är dessutom meningslöst, eftersom en modell med tio parametrar alltid är mindre.
2. **Svenska är fortfarande ett öppet mål, men inte orört.** Jag hittade två saker rapporterna missade (se avsnitt 3). Ingen av dem har en rapport eller en licens. Vi citerar dem i rapporten.
3. **Det ger en forskningsfråga på riktigt:** *Hur många parametrar behöver svenska jämfört med engelska för samma kvalitet?* Samma kod, samma data (översatt), två språk. Det är rapportens kärna och fungerar internationellt.

---

## 2. Vad de tre rapporterna säger, och vad jag tar från varje

| Rapport | Styrka | Svaghet | Tar med |
|---|---|---|---|
| **deep-research-report.md** (MIND/1M, engelska, under 1M parametrar) | Bäst på berättelse: "Minimum Viable Language Model"-kriterier, skalningsserie, *emergence-slidern* | Minst tillförlitlig. Flera påståenden gick inte att verifiera (StateHead 9,97M, "TinyLM 78K"). Under 1M parametrar på engelska ger inte sammanhängande text enligt både TinyStories-artikeln och Goedeckes mätningar | Slidern, kvalitetskriterierna, "Capability Cliff"-grafen |
| **Den minsta riktiga språkmodellen** (svensk TinyStories, Pytte 0,3M–40M) | Bäst källbelagd. Ärlig om risker, konkret om data, licenser och Mac-prestanda | Huvudpremissen "inget svenskt TinyStories finns" är fel (se avsnitt 3). Underskattar kostnad och tid för att generera 1–3 miljoner svenska sagor | Nästan hela planen: arkitektur, modellfamilj, faser, positionering, "försvarbara formuleringar" |
| **Världens Minsta LLM-Projekt** (Nano-LLM 15M, engelska, 4 veckor) | Praktisk om inferens: `run.c`, ONNX, INT4, WebGPU med WASM-fallback | Generisk, engelsk, föreslår rubriken "World's Smallest LLM" som inte håller. Tidsplanen på 4 veckor är optimistisk | Teknisk kedja för webbdemo och 4-filsstrukturen (`model.py`, `train.py`, `run.c`, `export.py`) |

Alla tre är överens om arkitekturen: dekoder-transformer i Llama-stil med RoPE, RMSNorm, SwiGLU, kausal multi-head attention, delade in-/utvikter, KV-cache, AdamW, egen BPE-tokeniserare med litet ordförråd. Det är samma grundarkitektur som i moderna språkmodeller, bara mycket mindre, och det ska vi säga rakt ut.

---

## 3. Vad jag verifierade (och vad som ändrade planen)

### Det finns redan svensk TinyStories-data, men ingen har berättat om den

- **`tdde19-llm-from-scratch/TinyStoriesSwe`** på Hugging Face: en maskinöversatt TinyStories till svenska, 2,46 GB text, publicerad december 2024 av en projektkurs vid Linköpings universitet (TDDE19). Datan saknar licens. Samma konto har 15 tränade svenska Llama-modeller (64–1024 dim, 2–8 lager) med ett ordförråd på 65 535 token, vilket i små modeller lägger det mesta av parametrarna på inbäddningen.
- **`qwrt/Swedish0.1M`** (juli 2026): en byte-nivå Qwen3-modell med 133 000 parametrar tränad på 7 miljarder tecken svenska. Författaren skriver själv att den "inte är så bra på att producera sammanhängande text".
- **`Dxniz/TinyStories-Multilingual`** (mars 2026, Apache-2.0): 15 688 sagor på 28 språk, alltså cirka 560 på svenska. För litet att träna på, men en bra validerings- och jämförelsekälla.

**Konsekvens:** Vi kan inte säga "första svenska TinyStories". Vi kan säga "den minsta *öppna, dokumenterade* modellen som skriver sammanhängande svenska, med definierat kvalitetsgolv". Och vi citerar Linköpings-datan och Swedish0.1M i rapporten. Det gör oss trovärdigare, inte svagare. Utan licens kan vi inte återanvända Linköpings-datan, men den är en bra extern kontroll: tränar vi på vårt eget dataset och mäter på deras valideringsmängd, visar vi att resultatet inte beror på vår data.

### Andra som gjort liknande

- **Language Model Builder** (Felix Rieseberg, juli 2026): gratis macOS-app som tränar en GPT i MLX utan kod.
- **microgpt** (Karpathy, februari 2026): hela algoritmen på 200 rader ren Python.
- **Sean Goedecke**: undersökte den starkaste modellen man kan träna på en laptop på fem minuter och landade på cirka 1,8–2M parametrar på TinyStories, perplexitet runt 9,6. Hans siffror ligger bakom tidsuppskattningarna nedan.

### Din Mac

| | |
|---|---|
| Dator | MacBook Pro, Apple M2 Pro, 10 kärnor, 32 GB enhetligt minne |
| Verktyg | Python 3.14 (plus 3.10–3.13 via `uv`), `uv` installerat. PyTorch 2.15 och MLX 0.32 har färdiga wheels för Python 3.14 på Apple Silicon. Varken torch eller MLX är installerat än |
| **Ledigt diskutrymme** | **1,1 GB av 460 GB** |

M2 Pro med 32 GB är gott och väl tillräckligt. Alla modellstorlekar upp till 40M ryms med marginal. Men **disken är ett stopp**: TinyStories är cirka 2 GB, ett svenskt dataset lika mycket, tokeniserad data, checkpoints och Python-miljö kräver minst 20–30 GB. Första åtgärden i projektet är att frigöra utrymme.

---

## 4. Förslaget: Pytte

*Arbetsnamn från den svenska rapporten (pytteliten). Byt gärna, men behåll något svenskt och uttalbart på engelska.*

### Modellfamiljen

Vi bygger inte en modell utan en skalningsserie, för det är kurvan som är resultatet. Varje storlek tränas på svenska och engelska med identisk kod.

| Namn | Form (ungefär) | Parametrar | Träningstid (uppskattning, M2 Pro) | Roll |
|---|---|---:|---|---|
| Pytte-0.3M | dim 64, 4–5 lager, vocab 2 048 | ~0,3M | 10–20 min | Golvet. Ryms på en mikrokontroller |
| Pytte-1M | dim 96–128, 4 lager | ~1M | 15–30 min | Var språket börjar uppstå |
| Pytte-3M | dim 128–192, 4 lager | ~3M | 30–60 min | Webbdemo som laddas direkt |
| Pytte-15M | dim 288, 6 lager, vocab 4 096 | ~15M | 3–8 h | **Huvudmodell.** Flytande sagor, int8 cirka 15 MB |
| Pytte-40M | dim 512, 8 lager | ~40M | 1–3 dygn | Flaggskepp med chattfinjustering. Valfri. |

Tiderna är extrapolerade från Goedeckes mätningar (1M: ~100 000 token/s, 2,6M: ~56 000 token/s på MacBook Pro) och Riesebergs (25M: ~16 000 token/s). M2 Pro är långsammare än M4 Pro, så räkna med att allt tar upp till dubbelt så lång tid. **Mät i fas 0.**

Vilken av dessa som blir "hero model" avgör experimenten, inte vi. Det är själva forskningen.

### Kvalitetsgolvet: det som gör "minst" försvarbart

Vi definierar i förväg vad "skriver sammanhängande svenska" betyder, så att påståendet blir reproducerbart och går att pröva. Förslag på kriterier, varje mätt på 200 slumpmässiga (inte handplockade) genereringar:

1. **Grammatik:** andel meningar utan böjnings- eller kongruensfel (en/ett, adjektivböjning, verbform). Bedöms av en LLM-domare med fast prompt, plus blindtest av 3–5 svensktalande personer.
2. **Sammanhang:** håller karaktärer, namn och plats genom hela sagan?
3. **Minimala par:** ger modellen högre sannolikhet åt "Hunden springer" än "Hunden springa"? En svensk mini-BLiMP på 100–200 par som vi publicerar.
4. **Nyhet:** andel genererade sagor som inte finns i träningsdatan (n-gram-överlapp).
5. **Perplexitet** på en held-out-mängd, plus **bits per byte** så att svenska och engelska går att jämföra trots olika tokeniserare.

Den minsta modellen som klarar tröskeln (till exempel 80 % på 1–2 och över 70 % på 3) räknas som den minsta som skriver sammanhängande svenska. Allt publiceras så att vem som helst kan kontrollera det.

### Demon

**Emergence-slidern** (från deep-research-rapporten, men på svenska). En webbsida, allt körs i besökarens webbläsare, ingen server:

```
0,3M ──── 1M ──── 3M ──── 15M ──── 40M
```

Användaren skriver "Det var en gång en älg som" och drar i reglaget. Vid 0,3M: ordsallad med svenska ord. Vid 1M: grammatiska fragment. Vid 3M: korta sagor med logiska fel. Vid 15M: en riktig saga. Man ser språk uppstå.

Ett tillägg:

- **Tokeniseringsvisning:** skriv "midsommarafton" och se hur svenska sammansättningar delas upp, med uppmärksamhetsmönster och nästa-ord-sannolikheter. Rapportens figurer kommer direkt härifrån.

### Fyra publika leverabler

1. **Koden** på GitHub: läsbar, få filer (`tokenizer.py`, `model.py`, `train.py`, `sample.py`, `eval.py`, `run.c`, `export.py`), `make train` och `make run`.
2. **Modellerna och datasetet** på Hugging Face med riktiga modellkort och datasetkort.
3. **Demon** på GitHub Pages / Hugging Face Space: slidern och visualiseringen.
4. **Rapporten** som PDF på GitHub och Hugging Face med DOI via Zenodo.

---

## 5. Data: det svåraste beslutet

Den svenska rapporten föreslår 1–3 miljoner nya svenska sagor. Det är för mycket för ett första varv. Goedecke nådde perplexitet 9,6 på 20M token, och TinyStories-artikelns 1M–33M-modeller tränades på ett dataset av cirka 470M token. **Mål för v1: 200 000–400 000 sagor, cirka 50–100M token.** Det räcker för 15M-modellen, och kurvan 0,3M–15M blir meningsfull. Mer kan genereras senare om 40M-modellen behöver det.

Två vägar, och jag föreslår att vi gör båda i liten skala i fas 0 och väljer:

| Väg | Fördel | Nackdel |
|---|---|---|
| **A. Översätt TinyStories** (licens CDLA-Sharing-1.0, kräver att det översatta delas under samma licens) | Exakt samma innehåll på svenska och engelska, vilket gör språkjämförelsen vetenskapligt ren | Översättningsartefakter ("modersflicka", amerikanska namn). Licensen är "viral" men det är okej för ett öppet dataset |
| **B. Generera nya svenska sagor** med TinyStories-metoden (slumpa substantiv, verb, adjektiv och ett drag som "dialog" eller "sorgligt slut") | Svensk prägel: midsommar, älgar, fika, skärgård. Egen licens | Språkjämförelsen blir svagare eftersom innehållet skiljer sig |

**Rekommendation:** B som huvuddataset ("SagoSvenska"), plus en mindre översatt delmängd av TinyStories (till exempel 50 000 sagor) enbart för den kontrollerade sv/en-jämförelsen i rapporten.

**Genereringsmodell:** välj en modell vars villkor uttryckligen tillåter att utdata används för att träna modeller. Öppna modeller under Apache-2.0 (till exempel Qwen-familjen) via en API-leverantör är den säkraste vägen. Stängda modellers användarvillkor kan begränsa träning på utdata, så kontrollera innan ni genererar en enda saga. Lokalt på M2 Pro går en 4–8B-modell för långsamt för hundratusentals sagor, men räcker för att testa promptar.

**Kostnad:** 300 000 sagor à cirka 250 token ut plus prompt motsvarar i storleksordningen 100–150M token. Med batch-rabatt hos en billig leverantör landar det troligen på några hundra till ett par tusen kronor. Räkna på aktuell prislista i fas 0. Det är projektets enda verkliga utgift. Elen för träningen är försumbar (en MacBook drar några tiotal watt; ett dygn är ungefär 1 kWh).

**Kvalitetskontroll:** språkidentifiering, deduplicering (TinyStories har själv cirka 15 % dubbletter och läckage mellan tränings- och valideringsmängd), filter för anglicismer, och ett manuellt stickprov på 200 sagor som du läser själv.

---

## 6. Plan

Totalt **7–9 veckor på kvällar och helger** för hela paketet. En snabbare väg finns i avsnitt 7.

### Fas 0: förstudie (denna vecka, 2–3 kvällar)

- Frigör **minst 30 GB** disk. Det är blockerande.
- Sätt upp miljö med `uv` (Python 3.12 eller 3.14), installera PyTorch och MLX.
- Mät token/s för en 2M-modell i MLX respektive PyTorch MPS på din Mac. Rapporterna är oense om vilken som är snabbast vid små modeller. Vi väljer efter mätning.
- Ladda ner engelska TinyStories (cirka 2 GB).
- Testa 20 svenska sagor var med väg A och B. Läs dem. Välj.
- Räkna på genereringskostnaden mot aktuell prislista.
- **Klart när:** vi har ett tal på token/s, ett valt ramverk, ett valt dataspår och plats på disken.

### Fas 1: engelsk prototyp (vecka 1–2)

- Hela kedjan på engelska TinyStories: egen BPE-tokeniserare (vocab 4 096), Llama-stil-modell, träningsloop med loggning och checkpoints, sampling med temperatur/top-k/top-p, KV-cache, utvärdering.
- Delmål: reproducera Goedeckes resultat, en ~2M-modell till perplexitet runt 10 på fem till tio minuter. Då vet vi att pipelinen är rätt.
- Träna första 15M på engelska över en natt.
- **Klart när:** `make train` ger en modell som skriver sammanhängande engelska sagor och vi har förlustkurvor.

### Fas 2: SagoSvenska (vecka 2–3, parallellt med fas 1)

- Genereringspipeline med variationsmekanism och svensk prägel.
- Generera 200 000–400 000 sagor plus 50 000 översatta TinyStories.
- Rensa, deduplicera, dela upp i träning/validering/test. Skriv datasetkortet medan du minns besluten.
- Litet SFT-set, 5 000–10 000 exempel: "Berätta en saga om…", "Vad händer sedan?"
- **Klart när:** datasetet ligger på Hugging Face (privat tills lansering) med kort och licens.

### Fas 3: skalningsserien (vecka 4–5)

- Träna 0,3M, 1M, 3M, 15M på svenska och engelska. 40M på svenska om tiden finns (kör över en helg med `caffeinate`).
- Bygg utvärderingssviten: LLM-domare med fast prompt, svensk mini-BLiMP, n-gram-nyhet, perplexitet och bits per byte.
- Hitta kvalitetsgolvet och "klippan" där svenskan blir sammanhängande.
- Ablationer på 3M-storleken (snabba att köra): RoPE mot inlärda positioner, SwiGLU mot GELU, djup mot bredd, vocab 2 048 mot 8 192, datamängd.
- Blindtest: fem personer bedömer 20 sagor var utan att veta modellstorlek.
- **Klart när:** vi har kurvan, tabellen och svaret på "hur många parametrar behöver svenska jämfört med engelska".

### Fas 4: chatt och portningar (vecka 6)

- SFT på 15M (och 40M om den finns) så den följer "Berätta en saga om en älg".
- `run.c`: en fil, inga beroenden, int8-export.
- Webbdemo: export till ONNX eller eget binärformat, inferens i JavaScript/WASM med WebGPU där det finns. Slidern och tokeniseringsvisningen.
- Valfritt: Pytte-0.3M på en ESP32 eller i en Game Boy-emulator.
- **Klart när:** demon fungerar på mobil utan installation och `./run pytte-15m.bin` skriver en saga i terminalen.

### Fas 5: rapport och visualisering (vecka 7–8)

- Rapport på engelska med svensk sammanfattning, struktur enligt den svenska rapportens avsnitt 7 (inklusive ett eget avsnitt om AI-assisterad utveckling: vad Claude Code gjorde, vad du beslutade, vad som gick fel).
- Arkitekturdiagram med verkliga dimensioner och en "parameterbudget"-stapel (andel inbäddning, uppmärksamhet, MLP). I små modeller är den fördelningen överraskande.
- Zenodo-DOI för rapport och dataset.
- Förhandsgranskning av 3–5 personer med ML-bakgrund. Fråga en svensk forskare om kommentar.
- **Klart när:** PDF, figurer och README är låsta.

---

## 7. Snabbspåret (3–4 veckor)

Om du vill ha något ute snabbt: fas 0, fas 1 bara till 2M-prototypen, fas 2 med 150 000 genererade sagor (ingen översättning), fas 3 bara 1M/3M/15M på svenska utan ablationer, fas 4 bara webbdemo med slidern, en kortare rapport. Du tappar sv/en-jämförelsen och chatten, men kärnan (svensk skalningsserie, slider, öppen data, ärlig rapport) är kvar. Resten kan komma som "v2".

---

## 8. Risker och hur vi hanterar dem

| Risk | Hantering |
|---|---|
| Disken är full | Frigör 30 GB innan något annat. Lägg datasets och checkpoints på en extern SSD om det blir trångt |
| Värmestrypning vid långa körningar | Nätadapter, `caffeinate`, logga token/s över tid, checkpoints var 10:e minut |
| Genereringsmodellens villkor förbjuder träning på utdata | Välj Apache-2.0-modell via API. Dokumentera valet i datasetkortet |
| Svenskan blir dålig i genererade sagor | Manuellt stickprov på 200 innan full körning. Filter för anglicismer. Hellre färre bra sagor än många dåliga |
| "Det finns redan" (Linköping, Swedish0.1M) | Citera dem i README och rapport dag 1. Visa skillnaden: licens, dokumentation, kvalitetsgolv, sv/en-jämförelse, demo |
| "Det är inte large" | Säg det först själv. SLM med LLM-arkitektur |
| Uppblåsta kvalitetspåståenden | Visa bara slumpmässiga utdata. Redovisa fel. Publicera domarprompten |
| Hur AI användes i arbetet | Redovisas öppet i rapporten: vad Claude Code gjorde och vad du beslutade |
| Rapporten har fel siffror | Räkna parametrar med och utan inbäddningar och ange båda. Mät energi med `powermetrics` innan ni påstår något om kronor |

---

## 9. Vad jag föreslår att vi gör härnäst

Fas 0 kan börja i den här sessionen så fort disken är löst:

1. Du frigör 30 GB (eller kopplar in en extern SSD och säger var).
2. Jag sätter upp miljön, skriver mätskriptet och mäter MLX mot MPS på en 2M-modell på din Mac.
3. Jag laddar ner TinyStories och bygger första tokeniseraren och träningsloopen.
4. Vi skriver tio svenska sagor med väg A och B och du läser dem.

Efter det har vi riktiga siffror i stället för uppskattningar, och planen ovan kan justeras därefter.

---

## 10. Fas 0: uppmätt 2026-11-06

**Beslut: noll kronor.** Inga API-köp, inga mejl, inget arXiv. Sagorna genereras lokalt med en öppen modell. Rapporten publiceras på GitHub/Hugging Face med Zenodo-DOI.

**Lagring.** Extern USB-disk (1 TB, APFS, cirka 95 MB/s skrivning, troligen snurrdisk), volymen `Modelltest`, 540 GB ledigt. Hela projektmiljön ligger i `/Volumes/Modelltest/pytte/` (`.venv`, `uv-cache`, `hf-cache`, `data`, `checkpoints`). Koden ligger i detta repo. Interna disken belastas inte.

**Miljö.** Python 3.12 via `uv`, torch 2.15.0 (MPS fungerar), mlx 0.32.4, mlx-lm 0.32.2, tokenizers 0.23.3, datasets 5.1.0.

**Träningshastighet, M2 Pro, identisk Llama-stil-modell (vocab 4 096, seq 256), slumpdata:**

| Modell | Parametrar | MLX (compiled) | MLX (utan compile) | PyTorch MPS |
|---|---:|---:|---:|---:|
| small: dim 128, 4 lager | 1,33M | **161 000 tok/s** | 134 000 | 99 000 |
| mid: dim 288, 6 lager | 7,16M | **44 500 tok/s** | 38 400 | 26 700 |

Batch 64 gav ingen vinst över 32. MLX är cirka 1,6 gånger snabbare än PyTorch MPS på båda storlekarna. **Beslut: MLX för träning.** En PyTorch-version kan behållas som läsbar referens.

Vad det betyder i praktiken: en 1,3M-modell ser 40M token på cirka 4 minuter. En 7M-modell ser 300M token (tre epoker på 100M) på cirka 2 timmar. En riktig 15M-modell bör hamna runt 25 000 tok/s, alltså 3–4 timmar för 300M token. Det är bättre än planens uppskattningar.

**Data.** TinyStories V2 (GPT-4-genererad) nedladdad: 2,1 GB träning, 21 MB validering.

**Lokal sagogenerering, uppmätt (M2 Pro, mlx-lm, 4-bit, cirka 150–250 token per saga):**

| Modell | Batch 1 | Batch 16 | Sagor/timme (batch 16) | Svenska | Licens för träning på utdata |
|---|---:|---:|---:|---|---|
| Qwen3-4B | 31 tok/s | 87 tok/s | ~1 700 | **Underkänd.** "skrök", "lånte", "fallde", "tårade", "ochtog", "sommar dag" | Apache-2.0, fri |
| Gemma 3 4B | 52 tok/s | 109 tok/s | ~1 900 | **Bra.** Nästan felfri, småfel som "sjöen", "Jag vänta" | Gemma Terms: en modell tränad på Gemma-genererad syntetisk data räknas som "Model Derivative" och måste bära Gemmas användarvillkor. Olämpligt. |
| Gemma 4 E4B | 43 tok/s | 93 tok/s | ~1 800 | **Underkänd.** "Byen", "Snöna", "Snöfaller", "ett litet drake", "De har små vänskap". Bättre vid temperatur 0,6 men fortfarande fel | **Apache-2.0** (bekräftat på google/gemma-4-E4B-it), inga begränsningar |
| **Gemma 4 12B-it (QAT)** | 11 tok/s | 34 tok/s (batch 16), **50 tok/s (batch 32)** | ~650 (batch 16), **~960 (batch 32)** | **Utmärkt.** Naturlig, felfri svenska: "Får jag låna dina skridskor?", "Ska vi hjälpas åt?" | **Apache-2.0**, inga begränsningar |

**Beslut: Gemma 4 12B-it (QAT, 4-bit), temperatur 0,6, batch 32.** Toppminne 20 GB av 32, så batchen kan inte ökas mycket mer. Cirka 960 sagor i timmen betyder 60 000 sagor på 2,5 dygn och 150 000 på knappt en vecka. Promptbearbetningen tar en tredjedel av tiden; cachning av den gemensamma systemprompten kan ge en bit till.

**Prompt-cache (15:00).** Systeminstruktionen är 109 av cirka 178 token per prompt och räknas nu bara en gång (KV-cache som kopieras per prompt, med fallback om en prompt inte delar prefixet). Mätt på samma 32 sagor och samma frö: 143 s utan cache, 109 s med, alltså **802 → 1 056 sagor i timmen (+32 %)**. I drift gav batch 32 + cache 1 180 sagor i timmen i fem batcher, sedan GPU out-of-memory (Metal) med 3,8 GB swap i bruk: MLX:s buffertcache växer över tid. Åtgärd: `mx.clear_cache()` efter varje batch, och batch 24 för marginal. Förväntad takt cirka 900–950 sagor i timmen, 60 000 på knappt tre dygn. Batch 32 kan provas igen när rensningen visat sig hålla.

Gemma 4 måste köras med tänkande avstängt (`enable_thinking=False` i chattmallen), annars går alla token till engelska resonemang. Observera också att mlx-community har både bas- och instruktionsmodeller med liknande namn: `gemma-4-12B-4bit` är basmodellen utan chattmall, `gemma-4-12B-it-qat-4bit` är rätt.

Gemma 4 26B-A4B (MoE, 4B aktiva parametrar, 15,6 GB) skulle kunna ge 12B-kvalitet i 4B-fart men stöds inte av mlx-lm 0.32.2. Värt att prova när stödet kommer.

Batch 32 var långsammare än batch 16 för Qwen3 (45 tok/s) men snabbare för Gemma 4 12B. Mät per modell.

**Barnsäkerhet i prompten.** Draget "ett sorgligt slut" gav en saga där ett barn drunknar under isen. Formulera om till "ett lite sorgligt men tryggt slut, ingen skadas" och lägg till ett filter i efterhand (död, drunkna, skada, blod).

Två lärdomar: (1) prompten måste slumpa huvudpersonens namn och miljö uttryckligen, annars heter var och varannan huvudperson "Lilla Leo"; (2) 150 000 sagor i cirka 1 900 sagor/timme tar ungefär 80 timmar, alltså tre till fyra dygn med datorn igång. Det är gratis men kräver att den står på. Alternativ: generera i omgångar över två veckor, eller börja med 50 000 sagor och träna de små modellerna medan resten genereras.

**Att komma ihåg vid implementation.** PyTorch-benchmarken divergerade (loss 28) eftersom `nn.Embedding` initieras med standardavvikelse 1 och vikterna delas med utgångslagret. Initiera inbäddningar med standardavvikelse 0,02 (GPT-2-stil) eller skala med 1/sqrt(dim).

## 11. Fas 1: första modellen tränad, 2026-11-06

Hela kedjan finns i `pytte/` (tokenizer, prepare_data, model, train, sample) och är verifierad: KV-cache ger identiska logits som full beräkning (avvikelse 4e-7), startförlust 8,34 mot teoretiska ln(4096) = 8,32, spara/ladda ger identiska vikter.

**Tokeniserare (engelska):** byte-nivå-BPE, 4 096 token, tränad på 288 MB TinyStories på 22 sekunder. TinyStories V2 tokeniserat: 552,6M träningstoken, 5,6M valideringstoken, 3,95 byte per token.

**Pytte-1M, fem minuter ensam på GPU:n (M2 Pro):**

| | |
|---|---|
| Parametrar | 1,33M (0,80M utan inbäddningar) |
| Steg / token | 2 885 steg, 23,6M token |
| Hastighet | 150 000 tok/s i stabilt läge, 78 700 i snitt inklusive kompilering och utvärdering |
| Valideringsförlust | 2,160 |
| **Perplexitet** | **8,67** (Goedecke: ~9,6 med 1,8M parametrar) |
| Bits per byte | 0,789 |
| Inställningar | batch 32 x 256, lr 2e-3 med 5 % uppvärmning och cosinus till 1e-4, AdamW (0,9/0,95, wd 0,1), gradientklipp 1,0 |

Exempel (slumpat, inte handplockat): *"Once upon a time, there was a little boy named Tim. Tim had a sad toy car. He was sad because he did not like his car. He wanted his toy car back."*

**Kernel panic 14:09.** Datorn startade om med "watchdog timeout: no checkins from watchdogd in 94 seconds" och lastsnitt 394. Orsak: 12B-genereringen (20 GB) plus utvärderingens n-gram-tabell (flera GB) fyllde minnet, och macOS kunde inte växla till disk eftersom den interna disken har 1,1 GB ledigt. Swap ligger alltid på den interna disken. Troligen samma orsak till hängningen vid Gemma 3-testet. Åtgärder: genereringen kör nu med batch 24 (~16 GB), utvärderingens nyhetsmått använder en kompakt numpy-tabell på 50 000 sagor (~80 MB), och **den interna disken måste få minst 15–20 GB ledigt för swap** innan fler tunga jobb körs parallellt.

**Lärdomar.** (1) Träning och sagogenerering samtidigt gör träningen tio gånger långsammare (12 000 tok/s). Genereringen äger GPU:n på nätterna, träning sker i pauser. STOP-filen pausar genereringen snyggt och den återupptas där den var. (2) Tidsbudget i stället för stegantal gör att inlärningsschemat blir rätt oavsett hastighet. (3) Perplexiteten var fortfarande på väg ner vid fem minuter, så 1M-modellen är inte mättad; en längre körning ger mer.

**Modellfamiljen som den är definierad i koden** (vocab 4 096 utom 0,3M med 2 048):

| Namn | dim | lager | huvuden | ff | Totalt | Utan inbäddningar | Andel inbäddning |
|---|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m | 64 | 4 | 4 (2 kv) | 192 | 0,33M | 0,20M | 40 % |
| pytte-1m | 128 | 4 | 4 | 352 | 1,33M | 0,80M | 39 % |
| pytte-2m | 160 | 5 | 5 | 448 | 2,24M | 1,59M | 29 % |
| pytte-3m | 192 | 6 | 6 | 512 | 3,44M | 2,66M | 23 % |
| pytte-7m | 288 | 6 | 6 | 768 | 7,16M | 5,98M | 16 % |
| pytte-15m | 384 | 8 | 6 | 1024 | 15,74M | 14,16M | 10 % |
| pytte-40m | 512 | 12 | 8 | 1408 | 40,65M | 38,55M | 5 % |

Andelen inbäddning är en bra figur för rapporten.

## 12. Fas 2: SagoSvenska klar, 2026-11-09

**60 000 sagor på 54 timmar.** Gemma 4 12B-it (QAT 4-bit) genererade 60 000 svenska sagor lokalt, torsdag 6/11 13:30 till måndag 9/11 06:24. Stabil takt 980 sagor/timme med batch 24 och prefix-cache. Totalt 6,6M ord, 110 ord per saga i snitt (58–192). Kostnad: noll kronor, drygt två dygns ström.

**Vad som krävdes för att hålla den igång i 54 timmar:**

| Problem | Åtgärd |
|---|---|
| Metal-OOM efter 2,7 h trots `mx.clear_cache()` per batch (buffertcachen kröp uppåt, batchtiderna gick från 83 till 131 s) | `mx.set_cache_limit(2 GB)` och `mx.set_memory_limit(22 GB)`. Därefter konstant 10,2 GB aktivt minne i 50 timmar. |
| En krasch kostar hela körningen | `data/gen_loop.sh`: startar om generatorn efter 30 s, den återupptar från sparat antal |
| 30 W-laddare: batteriet sjönk 1 %/min under last trots nätsladd | Batterivakt i loopen: PAUSE-fil vid 30 %, återupptar vid 90 % eller när en laddare ≥ 60 W känns av. Slog till en gång (söndag kväll, laddaren urdragen) och återupptog själv när 65 W-laddaren kom tillbaka. |

65 W räcker för M2 Pro under GPU-last (batteriet laddar långsamt), 30 W och 25 W gör det inte. Locket måste vara öppet; skärmlås och skärmsläckning är ofarliga.

**Flaggade sagor:** 601 av 60 601 (1 %): 573 "osäkra" och 28 avhuggna. De osäkra var till 80 % falska träffar på "slår" (hjärtat slår, slår sig ner) och "kista" (skattkista). Rensningen prövar dem igen med ett snävare filter.

**Rensning (`data/clean_stories.py`, 34 s):** 60 573 in, 60 431 kvar. Bort: 133 osäkra, 7 obalanserade citattecken, 2 med engelska ord. **Noll exakta och noll nästan-dubbletter** (MinHash, Jaccard ≥ 0,7). Dedup-logiken verifierades med planterade dubbletter. Att 60 000 sagor från samma modell inte innehåller en enda nästan-dubblett beror på att varje prompt slumpar namn, plats, sak, handling, egenskap och längd, ~10 miljoner kombinationer.

**Ordförråd:** 11 161 unika ord, 532 ord täcker 90 % av texten. Det är TinyStories-effekten på svenska: ett litet ordförråd som en liten modell kan lära sig helt. Uppdelning: 59 195 träning / 619 validering / 617 test (deterministiskt på texthash).

**Tokeniserare och token:** byte-BPE 4 096 tränad på 2,7 s. 8,14M träningstoken, 85k valideringstoken. Det är 68 gånger mindre än TinyStories (553M), så 1M-modellen ser varje saga ~4 gånger på 8 minuter och 15M-modellen börjar memorera. Bästa valideringscheckpoint sparas, så memoreringen syns i kurvan utan att förstöra modellen. För 0,3M används vocab 2 048 (8,68M token).

**Träningsbudget per modell (samma för svenska och engelska, `scripts/train_family.sh`):** 0,3M 5 min, 1M 8 min, 2M 10 min, 3M 12 min, 7M 20 min, 15M 30 min. lr 2e-3 upp till 3M, 1e-3 för 7M och 15M. Resultaten fylls i i avsnitt 13 när körningen är klar.

## 13. Fas 3: den svenska modellfamiljen, 2026-11-09

Sex modeller tränade i följd på GPU:n med `scripts/train_family.sh sv` (v1.0 07:52–09:34; **tabellen visar v1.1**, omtränad 21:50–23:42 samma dag på det rensade datat efter `<image|>`-buggen, se avsnitt 15, där också v1.0-siffrorna står). Alla på SagoSvenska (8,1M token), batch 32 × 256, AdamW, cosinus-schema på tidsbudget, bästa valideringscheckpoint sparad. Utvärdering med 100 slumpade sagor per modell (temp 0,8, top-p 0,95), inget handplockat. Minimala par: v2 med 200 längdmatchade par (omkörd 13:29–13:44, se Fynd 4). Domarkolumnen (Gemma 4 12B betygsätter grammatik/sammanhang/kreativitet 1–10 på 50 sagor per modell) fylls i avsnitt 14.

| Modell | Param | Min | Mtoken | Epoker | Ppl | Bits/byte | Minimala par | Nya 7-gram | Upprepn. 4-gram | Slutade själv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m-sv | 0.33M | 5 | 82 | 9.4 | 4.69 | 0.538 | 67.5% | 50% | 1.9% | 86% |
| pytte-1m-sv | 1.33M | 8 | 80 | 9.8 | 3.76 | 0.433 | 76.0% | 44% | 1.3% | 95% |
| pytte-2m-sv | 2.24M | 10 | 60 | 7.4 | 3.57 | 0.417 | 77.0% | 45% | 1.4% | 93% |
| pytte-3m-sv | 3.44M | 12 | 54 | 6.6 | 3.49 | 0.409 | 79.0% | 45% | 1.1% | 97% |
| pytte-7m-sv | 7.16M | 20 | 55 | 6.8 | 3.43 | 0.403 | 80.0% | 44% | 1.1% | 93% |
| pytte-15m-sv | 15.74M | 30 | 52 | 6.4 | 3.45 | 0.405 | 77.0% | 44% | 1.2% | 98% |

Perplexitet för 0,3M är inte jämförbar med de andra (vocab 2 048 mot 4 096); bits per byte är det jämförbara måttet. **Rättelse 13:30:** den första versionen av tabellen angav bits/byte 0,95–1,20. Det var en bugg i `train.py`: byte per token räknades på de första 200 000 valideringstoken, men den svenska valideringsmängden har bara 84 893, så kvoten blev 2,36 gånger för låg (1,87 i stället för 4,40 byte/token). `pytte.eval` räknade rätt hela tiden; buggen är fixad och alla `summary.json` korrigerade. De engelska siffrorna påverkades inte (5,6M valideringstoken).

**Fynd 1: redan 0,3M skriver svenska.** Pytte-0.3M (328 000 parametrar, 197 000 utan inbäddningar, fem minuters träning) producerar grammatiskt korrekta, sammanhängande meningar: *"Elsa går ut på en stor äng. Det finns många fina blommor där. Blommorna är gula och vita. Elsa ser en liten fjäril. Fjärilen är gul och blå. Den flyger högt upp i luften."* (det första av de 100 utvärderingsexemplen för v1.1, inte utvalt). 86 % av sagorna avslutas av modellen själv. Det är mindre än allt vi hittade som påstås skriva svenska (qwrt/Swedish0.1M på 133k var inte begriplig), och 0,3M är inte golvet: vi har inte provat mindre.

**Fynd 2: klippan ligger under 1M, platån börjar vid 3M.** Bits per byte: 0,54 → 0,43 → 0,42 → 0,41 → 0,40 → 0,41 för 0,3M → 15M. Det stora språnget är 0,3M → 1M. Från 3M och uppåt är alla på samma valideringsförlust (1,23–1,25) och ingen av dem överanpassar inom budgeten: 15M:s valideringskurva planar ut vid steg ~5 000 men vänder inte uppåt. Tolkning: med 8M token (modellerna ser datat 6–10 gånger) och en lågtemperatur-generator (temp 0,6) är datats egen oförutsägbarhet nådd runt 3M parametrar. Mer modell ger inget, mer och mer varierat data skulle göra det. Det är ett resultat i sig för rapporten: **datamängden, inte modellstorleken, sätter taket**. Avsnitt 14 visar att de engelska tvillingarna på 553M token fortsätter förbättras hela vägen till 15M.

**Fynd 3: datat är mallartat.** 59 224 sagor har bara 9 149 olika första meningar. "Det är en fin dag." förekommer i 9 743 sagor, "Det är vinter i fjällen." inleder 994. Det är Gemma vid temp 0,6. Det förklarar varför nya 7-gram ligger på 44–50 % (engelska 1M på TinyStories: 55 %) och varför 3M och 7M öppnar identiskt med samma slumpfrö. Inför publicering bör en SagoSvenska v2 genereras med temp 0,8–0,9 och fler påtvingade öppningar; kostnaden är ytterligare ~50 timmar GPU.

**Fynd 4: grammatiktestet (v2, 200 par) är platt över storlekarna, och pluralis är den svaga punkten.** Det första testet (63 par) var för litet och jämförde i pluraliskategorin meningar med olika antal token. Det byggdes om 2026-11-09 till 200 par i tio kategorier, 20 per kategori, lika många ord i båda meningarna och nästan alltid lika många token (161 av 200 par har exakt lika många token med tok-sv-4096). Resultat per kategori:

| Kategori | 0.3m | 1m | 2m | 3m | 7m | 15m |
|---|---:|---:|---:|---:|---:|---:|
| genus (en/ett) | 100% | 100% | 95% | 100% | 100% | 100% |
| bestämdhet | 65% | 90% | 85% | 100% | 80% | 90% |
| modal + infinitiv | 90% | 90% | 95% | 95% | 95% | 90% |
| preteritum | 75% | 85% | 85% | 90% | 95% | 90% |
| presens | 80% | 80% | 85% | 65% | 90% | 85% |
| ordföljd (V2) | 70% | 75% | 75% | 80% | 70% | 75% |
| pronomen (kasus) | 50% | 85% | 70% | 80% | 80% | 80% |
| bisatsnegation | 45% | 65% | 50% | 75% | 85% | 40% |
| adjektivkongruens | 65% | 55% | 75% | 55% | 60% | 70% |
| pluralis | 35% | 35% | 55% | 50% | 45% | 50% |
| **alla 200** | **67.5%** | **76.0%** | **77.0%** | **79.0%** | **80.0%** | **77.0%** |
| per token (längdnormerat) | 69.5% | 77.0% | 77.5% | 80.5% | 80.0% | 80.0% |

Tabellen är v1.1. v1.0 (samma frö, samma tokenizerbalans, data med `<image|`-skräpet) gav 77,5 / 79,5 / 81,0 / 79,0 / 80,0 / 82,5 %, alltså upp till tio procentenheter högre för samma konfiguration; se avsnitt 15 om vad det säger om testets brus.

Tre saker att ta med sig. (a) Genus, modalverb och tempus sitter redan vid 0,3M–1M, bestämdhet från 1M: modellerna vet att det heter *ett hus* och *den röda bollen*. (b) Totalen ligger på 76–80 % från 1M till 15M och rangordnar inte storlekarna; det bekräftar platån i Fynd 2 från ett annat håll. Bruset är dock större än de ±3 procentenheter som binomialfördelningen ger vid 200 par: mellan v1.0 och v1.1 flyttade sig 0,3M tio enheter och 15M fem, med samma frö och nästan samma data, och tre frön på 0,3M gav 67,5–80,0 % vid identisk bits/byte (avsnitt 15). Från 1M är testet stabilt inom en enhet över frön. Den enda säkra storleksskillnaden är att 0,3M ligger under de andra. (c) Pluralis ligger på slumpnivå i alla storlekar (35–55 %); typiska par är *"Tre pojkar spelar boll"* mot *"Tre pojken spelar boll"* och *"Barnen är glada"* mot *"Barnen är glad"*. Bestämd singular är den överlägset vanligaste nominalformen i sagorna (en huvudperson, en hund, en boll), så modellerna har sett få pluraler; det är ett dataproblem som SagoSvenska v2 bör adressera (fler sagor med flera figurer och föremål). Det är också en bra sak att visa i demon: en modell som skriver flytande svenska men inte kan böja i plural.

**Hastigheter (M2 Pro, MLX, ensam på GPU:n):** 0,3M 274k tok/s, 1M ~170k, 2M ~100k, 3M ~78k, 7M ~47k, 15M ~27k i snitt inklusive utvärderingar.

De engelska tvillingarna med exakt samma budgetar på TinyStories finns i avsnitt 14.

## 14. Fas 3b: tvillingjämförelsen svenska/engelska, 2026-11-09

Samma kod, samma sex konfigurationer, samma tidsbudgetar (5/8/10/12/20/30 min), samma batch och schema. Enda skillnaden är datat: SagoSvenska (8,1M token, 60k sagor, Gemma vid temp 0,6) mot TinyStories V2 (553M token, 2,7M sagor, GPT-4). Senare på kvällen (18:56–19:55) tränades en tredje familj, **en8m**: samma kod och budgetar på ett slumpurval av TinyStories V2 med exakt SagoSvenskas token-antal (40 027 sagor, 8,14M token, 423 valideringssagor), så att engelskan får lika lite data och lika många epoker som svenskan. Den raden i tabellen är kontrollen för allt som sägs om data kontra språk nedan. De engelska kördes 11:31–13:27 med `scripts/train_family.sh en`. Två tidigare engelska omgångar kasserades: den första lästes från en memmap på USB-disken och blev I/O-bunden (24–39k tok/s mot 170k), den andra förlorade fem minuter på att datorn sov mitt i budgeten (`time.time()` i stället för `time.monotonic()`). Båda felen är rättade i `train.py`. Tokenantalen per budget matchar nu inom 5 % mellan språken.

Minimala par: 200 längdmatchade par per språk, tio kategorier (engelska v2 byggd 13:45 efter svensk förebild). Domaren är Gemma 4 12B med fast prompt och temp 0, betyg 1–10 för grammatik/sammanhang/kreativitet på 50 slumpade sagor per modell (samma sagor som genereringsstatistiken; alla 600 svar parsades). Bpb @N = bits/byte vid N miljoner sedda token, avläst ur träningsloggen, så att modellerna också kan jämföras vid lika mycket data i stället för lika mycket tid. De svenska raderna är v1.1 (avsnitt 15).

| Modell | Språk | Mtoken | Epoker | Bits/byte | Bpb @10M | Bpb @20M | Bpb @40M | Minimala par | Per token | Nya 7-gram | Domare G/S/K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m | sv | 82 | 9.4 | 0.538 | 0.687 | 0.616 | 0.577 | 67.5% | 69.5% | 50% | 8.7/5.3/2.6 |
| pytte-0.3m | en | 86 | 0.16 | 0.842 | 1.029 | 0.945 | 0.900 | 83.5% | 85.0% | 61% | 7.2/2.9/2.3 |
| pytte-0.3m | en8m | 79 | 9.0 | 0.859 | 1.045 | 0.966 | 0.906 | 79.0% | 84.5% | 63% | 7.2/2.8/2.2 |
| pytte-1m | sv | 80 | 9.8 | 0.433 | 0.552 | 0.503 | 0.465 | 76.0% | 77.0% | 44% | 9.5/8.8/3.0 |
| pytte-1m | en | 81 | 0.15 | 0.650 | 0.840 | 0.776 | 0.711 | 90.0% | 88.0% | 55% | 9.3/4.5/2.5 |
| pytte-1m | en8m | 74 | 9.1 | 0.690 | 0.859 | 0.786 | 0.742 | 88.0% | 85.5% | 57% | 8.8/4.9/2.7 |
| pytte-2m | sv | 60 | 7.4 | 0.417 | 0.526 | 0.479 | 0.435 | 77.0% | 77.5% | 45% | 9.5/9.0/3.0 |
| pytte-2m | en | 60 | 0.11 | 0.634 | 0.802 | 0.737 | 0.661 | 90.5% | 89.5% | 56% | 9.3/4.9/2.5 |
| pytte-2m | en8m | 59 | 7.3 | 0.657 | 0.829 | 0.753 | 0.695 | 91.5% | 88.5% | 57% | 9.0/4.9/2.6 |
| pytte-3m | sv | 54 | 6.6 | 0.409 | 0.513 | 0.465 | 0.420 | 79.0% | 80.5% | 45% | 9.7/9.1/3.0 |
| pytte-3m | en | 56 | 0.10 | 0.606 | 0.782 | 0.716 | 0.634 | 91.5% | 90.5% | 55% | 9.5/6.1/2.8 |
| pytte-3m | en8m | 52 | 6.4 | 0.643 | 0.802 | 0.719 | 0.666 | 90.0% | 91.0% | 55% | 9.6/5.9/2.9 |
| pytte-7m | sv | 55 | 6.8 | 0.403 | 0.496 | 0.449 | 0.412 | 80.0% | 80.0% | 44% | 9.5/9.1/3.0 |
| pytte-7m | en | 57 | 0.10 | 0.576 | 0.760 | 0.685 | 0.603 | 90.5% | 90.0% | 54% | 9.5/5.6/2.7 |
| pytte-7m | en8m | 59 | 7.3 | 0.623 | 0.781 | 0.697 | 0.653 | 92.5% | 92.5% | 55% | 9.2/5.1/2.8 |
| pytte-15m | sv | 52 | 6.4 | 0.405 | 0.486 | 0.438 | 0.405 | 77.0% | 80.0% | 44% | 9.7/9.4/3.1 |
| pytte-15m | en | 51 | 0.09 | 0.551 | 0.733 | 0.656 | 0.573 | 94.0% | 92.5% | 53% | 9.8/7.3/3.0 |
| pytte-15m | en8m | 53 | 6.5 | 0.620 | 0.755 | 0.676 | 0.638 | 93.0% | 92.5% | 54% | 9.5/6.0/2.8 |

**Fynd 5: bits/byte går inte att jämföra rakt över språken, men kurvornas form kan.** Svenskan ligger lägre överallt (0,40–0,54 mot 0,55–0,84). Det betyder inte att svenska är lättare: SagoSvenska är genererad vid temp 0,6 och mycket mallartad (Fynd 3), och de svenska modellerna ser varje token 6–10 gånger medan de engelska aldrig ser samma token två gånger (0,09–0,16 epoker). Det som går att jämföra är hur kurvan beter sig med storleken. Engelska förbättras monotont hela vägen: 0,84 → 0,65 → 0,63 → 0,61 → 0,58 → 0,55, och vid lika många sedda token (40M) 0,90 → 0,57. Svenska stannar vid 3M: 0,54 → 0,43 → 0,42 → 0,41 → 0,40 → 0,41. Samma arkitektur, samma budget, olika data, och bara det stora datasetet fortsätter belöna fler parametrar. Det är tvillingens bekräftelse av Fynd 2: **taket i den svenska familjen sitter i datat**.

**Fynd 6: grammatiktestet skiljer språken åt och växer bara på engelska.** Engelska 83,5 → 94,0 % från 0,3M till 15M; svenska 67,5 → 77,0 %, platt från 1M. Men svagheterna är nästan desamma: det som fallerar i båda språken är **numeruskongruens**. Svenska: pluralis 35–55 % och adjektivkongruens 55–75 % (bisatsnegation hoppar 40–85 % och är den kategori som är känsligast för brus). Engelska: subjekt–verb-kongruens (*the dog run*) 55–85 % och reflexiver 65–80 %, medan ordföljd, negation, preteritum och modalverb sitter på 100 % redan vid 0,3M. Det engelska testet är troligen lättare (fler kategorier där den felaktiga formen är ett icke-ord som *goed*), så nivåskillnaden ska inte övertolkas; formen på kurvorna och kategorimönstret är det som håller.

**Fynd 7: domaren ger svenskan högre sammanhang, lägre nyhet.** Gemma betygsätter grammatiken högt i båda språken (7–9 av 10 redan vid 0,3M, 9,3–9,8 från 1M), men sammanhanget skiljer: svenska 5,3 → 8,8 redan vid 1M och sedan 9,0–9,4, engelska 2,9 → 7,3 med stadig ökning hela vägen till 15M. De svenska sagorna är korta, enkla och följer samma mall, så de hänger ihop; de engelska är längre och friare och tappar oftare tråden, men blir bättre med varje storlek. Vid 0,3M är det bara den svenska modellen som domaren anser hålla ihop en saga (5,3 mot 2,9), vilket stöder att "skriver sammanhängande svenska vid 0,3M" är ett rimligt påstående medan motsvarande inte gäller engelska på TinyStories. Kreativiteten är låg i båda (2,3–3,1): det här är modeller som återger ett sagoschema, inte hittar på. Nya 7-gram säger samma sak från ett annat håll: 44–50 % svenska mot 53–61 % engelska, de svenska modellerna återanvänder längre bitar ur träningsdatat.

**Vad tvillingen sv/en kan och inte kan säga.** Frågan i README, *hur många parametrar behöver svenska jämfört med engelska?*, går inte att besvara med bara de två första dataseten: de skiljer sig i storlek (68×), temperatur, källmodell och längd, och varje skillnad i kurvorna kan bero på datat lika gärna som på språket. Det som går att säga: (1) sammanhängande sagor dyker upp vid samma storlek, 0,3M, i båda språken, och klippan 0,3M → 1M finns i båda; (2) numeruskongruens är den sista grammatiska färdigheten i båda. Påståendet (3) som stod här tidigare, *ett dataset på 8M token sätter taket runt 3M parametrar oavsett språk*, var en gissning och visade sig fel: det testades samma kväll med en8m, se Fynd 8.

**Fynd 8: den storleksmatchade engelska familjen (en8m) skiljer data från språk.** Fyra saker faller ut.

1. **Taket sitter i datamängden, men det ligger högre för engelskan.** Med 8M engelska token går bits/byte 0,86 → 0,69 → 0,66 → 0,64 → 0,62 → 0,62: kurvan planar ut vid 7M parametrar (7M → 15M ger 0,003). Svenskan på lika mycket data planar ut vid 3M. Den fulla engelska familjen (553M token) fortsätter till 15M. Avståndet mellan en och en8m växer med storleken: 0,02 vid 0,3M, 0,04 vid 1–3M, 0,05 vid 7M, 0,07 vid 15M. Små modeller märker inte om datasetet är 8M eller 553M token; stora gör det. Det här är den rena data­skalningseffekten, mätt på samma maskin med samma kod.

2. **Svenskans låga bits/byte är datats, inte språkets.** Vid lika datamängd och lika epoker ligger engelskan på 0,62–0,86 och svenskan på 0,40–0,54. De 0,2 bits/byte är hur mycket mer förutsägbar SagoSvenska (Gemma vid temp 0,6, mallar) är än TinyStories (GPT-4, friare). Argumentet för SagoSvenska v2 vid högre temperatur är därmed kvantifierat.

3. **Grammatik kräver inte 553M token.** en8m klarar minimala par lika bra som den fulla familjen (79–93 % mot 84–94 %) och per token är de identiska från 3M. Den svenska familjens 68–80 % beror alltså inte på liten datamängd: engelska på samma 8M token når 93 %. Det som återstår är språket (svensk numeruskongruens är svårare än engelsk) eller testets svårighetsgrad (Fynd 6). Att skilja dem åt kräver ett svårare engelskt test eller ett tredje språk, och det är ett bra kapitel i rapporten snarare än något som behöver lösas nu.

4. **Nyhet och sammanhang följer också datat.** Nya 7-gram för en8m är 54–63 %, samma som en (53–61 %) trots 6–9 epoker. Att se datat många gånger ger alltså inte i sig mer kopiering; de svenska modellernas 44–50 % är SagoSvenskas mallar. Domarens sammanhang för en8m går 2,8 → 6,0, nästan identiskt med en upp till 3M (5,9 mot 6,1) och sedan efter (5,1/6,0 mot 5,6/7,3), precis där bits/byte-kurvorna går isär. Grammatikbetyget är lika (7,2 → 9,5). Slutar naturligt: 55–88 % för en8m, 67–83 % för en, 86–98 % för sv.

Slutsats för rapporten: påståendet *pytte-0.3m skriver sammanhängande svenska* står sig (domarens sammanhang 5,3 mot 2,8 för engelska på lika mycket data), men orsaken är att SagoSvenska är enklare och mer förutsägbar text, inte att svenska är lättare än engelska. Den ärliga formuleringen blir: *på ett enkelt, mallartat sagospråk räcker 0,3M parametrar, på friare text behövs 1M för samma sak*. SagoSvenska v2 vid högre temperatur får visa var svenskan hamnar när mallarna är borta; en8m-familjen är då referensen den ska jämföras med.

**Rättade mätfel under dagen** (alla i `pytte/train.py`, alla upptäckta genom att siffror inte stämde med varandra): I/O-bunden memmap, väggklocka under sömn, byte per token delat med fel antal. Lärdom för rapporten: varje tal som publiceras ska ha räknats på två sätt.

## 15. Fas 3c: pilot för SagoSvenska v2, och en bugg i v1, 2026-11-09 kväll

**Pilot.** Innan 50 timmar läggs på v2 genererades 500 sagor vardera med två skrivare vid temperatur 0,85, samma recept som v1: Gemma 4 12B (QAT 4-bit, v1-skrivaren) och Gemma 4 26B A4B (expertblandning, 4B aktiva parametrar, QAT 4-bit, 15,6 GB, nedladdad till externdisken). `scripts/pilot_stats.py` jämför set av sagor utan GPU, på slumpurval av 500.

| Mått | v1 (12B, temp 0,6) | 12B, temp 0,85 | 26B A4B, temp 0,85 |
|---|---:|---:|---:|
| Sagor per timme på M2 Pro | 980 | 1 136 | 1 710 |
| Sagor med läckt specialtoken | 12 % | 17 % | 0 % |
| Ord per saga | 110 | 112 | 118 |
| Unika ord (500 sagor) | 2 045 | 2 285 | 2 155 |
| Andel engångsord | 30 % | 34 % | 29 % |
| Olika öppningar (fyra första orden) | 81 % | 75 % | 84 % |
| Börjar "Namn bor/är/går ..." | 82 % | 80 % | – |
| 4-gram som återkommer i andra sagor | 29 % | 27 % | 27 % |
| Sagor med dialog | 26 % | 29 % | 41 % |
| Domare 12B: grammatik / sammanhang / kreativitet | 9,9 / 9,9 / 3,1 | 9,7 / 9,9 / 3,2 | 9,9 / 9,8 / 3,2 |
| Domare 26B A4B: grammatik / sammanhang / kreativitet | 9,6 / 9,4 / 3,7 | 9,3 / 9,5 / 4,0 | 9,8 / 9,6 / 3,9 |
| Språkfel per saga (12B-domare / 26B-domare) | 0,11 / 0,38 | 0,30 / 0,57 | 0,08 / 0,20 |

Domarraderna: 100 slumpade sagor per set (samma 100 för båda domarna), temp 0, kördes 00:16–01:04 när GPU:n var ledig. v1-setet är råsagorna inklusive `<image|`-skräpet.

**Fynd 9: temperaturen bryter inte mallen; receptet gör det.** Högre temperatur gav ett något rikare ordförråd men nästan oförändrad återanvändning mellan sagor (29 → 27 %) och samma andel sagor som öppnar med "Namn bor/är i ..." (82 → 80 %). Mallen kommer från prompten: varje saga får *namn, plats, substantiv, verb, adjektiv, drag* och Gemma löser uppgiften på samma sätt varje gång. v2 måste ändra receptet: slumpad öppningstyp (dialog, ljud, fråga, väder, handling mitt i), slumpat tempus, fler och mer olika drag, varierad längd. Det är en större hävstång än både temperatur och skrivarens storlek.

**Fynd 10: 26B A4B är den bättre skrivaren på den här maskinen.** 1,5 gånger snabbare än 12B eftersom bara 4B parametrar är aktiva per token, noll läckta specialtoken på 500 sagor, mer dialog och ibland preteritum (12B skriver nästan bara presens). Svenskan är på samma barnboksnivå och inte felfri i någon av dem ("det är läskig"). Båda domarna är eniga om rangordningen på språkfel: 26B A4B gör minst (0,08–0,20 per saga), 12B vid temp 0,85 flest (0,30–0,57), v1 däremellan, och 26B A4B får högst grammatikbetyg av båda domarna, även av 12B som då betygsätter en annan modell än sig själv. Sammanhanget är 9,4–9,9 överallt (korta mallsagor hänger ihop) och kreativiteten 3,1–4,0 överallt, oavsett skrivare och temperatur: det är receptet som sätter kreativitetstaket, vilket är Fynd 9 från domarens håll. Två förbehåll: 12B-domaren ger v1 9,9 i grammatik fast 12 % av sagorna innehåller `<image|`-skräp, så domaren ser förbi tokenskräp och ska inte användas för att hitta sådant; och 26B-domaren ger sin egen skrivare högst betyg, så självpreferens kan ligga bakom en tiondel eller två. Beslutet påverkas inte: v2 skrivs med 26B A4B. Den täta 31B valdes bort: 2,5 gånger dyrare per token och 17–18 GB vikter på en 32 GB-maskin som redan fått kärnpanik av minnesbrist.

**Buggen: `<image|>` i träningsdatat.** Den 4-bitars Gemma 4 12B spottar ibland ur sig sitt eget specialtoken för bildslut, `<image|>`, mitt i ett ord ("gung<image|>a", "l<image|>åda") eller direkt efter en punkt. Det hände i 12 % av v1:s råsagor (6 279 av 52 063 i huvudfilen; 4 708 mitt i ord, 1 796 efter skiljetecken). Rensningens markdown-regex tog bara bort tecknet `>`, så 7 097 träningssagor (12 %) innehöll strängen `<image|`, tokenizern lärde sig ett eget token `image`, och 5–14 % av de svenska Pytte-modellernas sagor innehöll skräpet (0,3M: 5 av 100, 15M: 14 av 100). Avsnitt 13–14 visar nu v1.1-siffrorna; jämförelsen mot v1.0 står nedan.

Rättat på tre ställen samma kväll. (1) `data/gen_stories.py`: en logit-processor sätter alla specialtoken utom `<eos>` och turslutet `<turn|>` till −∞ under samplingen (23 token spärrade, verifierat mot tokenizern), och en saga där något ändå läcker går till granskningsfilen med orsak `special_token`. (2) `data/clean_stories.py`: `repair_special_tokens()` tar bort tokenet och fogar ihop ordet, eller sätter ett mellanslag efter skiljetecken; en saga med `<` eller `|` kvar kastas. Omrensning av v1: 8 835 sagor reparerade, 60 426 behållna (v1.0: 60 431), noll `<` i träningsdatat, 11 619 unika ord. (3) v1.0-artefakterna (rensat data, tokenizers, token-filer, sex svenska checkpoints) flyttades till `/Volumes/Modelltest/pytte/archive-v1.0-imagebug/`, och den svenska familjen tränas om på v1.1 med `scripts/redo_sv_v1_1.sh` (samma budgetar, utvärdering, domare). Valideringssplitten ändras marginellt (599 mot 619 sagor) eftersom den hashas på innehåll, så v1.1-siffrorna ersätter v1.0 i stället för att jämföras rakt mot dem. Regel för framtiden: grep efter `<` i allt nytt data innan träning.

**v1.1-resultat (omträning 21:50–23:42, domare 23:42–00:15).** Samma sex budgetar, samma frö (0), samma tokenizerbalans i grammatiktestet (161 av 200 par lika långa med både gamla och nya tok-sv-4096). Antal steg per modell inom 2–7 % av v1.0. Inget av de 600 utvärderingsexemplen innehåller skräpet längre (v1.0: 5–14 % av sagorna).

| Modell | Bits/byte | Minimala par | Nya 7-gram | Domare gram./samm. | Slutade själv |
|---|---:|---:|---:|---:|---:|
| 0,3M | 0,543 → 0,538 | 77,5 → 67,5 % | 52 → 50 % | 8,6/5,6 → 8,7/5,3 | 88 → 86 % |
| 1M | 0,432 → 0,433 | 79,5 → 76,0 % | 43 → 44 % | 9,6/8,9 → 9,5/8,8 | 93 → 95 % |
| 2M | 0,418 → 0,417 | 81,0 → 77,0 % | 45 → 45 % | 9,6/8,6 → 9,5/9,0 | 95 → 93 % |
| 3M | 0,406 → 0,409 | 79,0 → 79,0 % | 45 → 45 % | 9,5/8,6 → 9,7/9,1 | 89 → 97 % |
| 7M | 0,403 → 0,403 | 80,0 → 80,0 % | 45 → 44 % | 9,4/8,7 → 9,5/9,1 | 94 → 93 % |
| 15M | 0,406 → 0,405 | 82,5 → 77,0 % | 43 → 44 % | 9,6/9,0 → 9,7/9,4 | 95 → 98 % |

**Fynd 11: skräpet kostade inget mätbart, men grammatiktestet är brusigare än vi trodde.** Bits/byte ändras med högst 0,005, nya 7-gram med högst två enheter och domarens betyg med 0,1–0,5 (sammanhanget något upp från 2M). Alla slutsatser i Fynd 1–3 och 5–8 står. Det som flyttade sig är minimala par: totalen gick ner för 0,3M (−10), 1M (−3,5), 2M (−4) och 15M (−5,5) och stod still för 3M och 7M. Det är inte tokeniseringen (balansen är identisk) och inte fröet (samma). Per kategori sitter rörelsen i bisatsnegation (0,3M 80 → 45 %, 15M 80 → 40 %), pronomen (0,3M 80 → 50 %) och bestämdhet (0,3M 80 → 65 %), medan genus, modalverb och tempus är oförändrade. Kategorierna som hoppar är de där modellen väljer mellan två ordföljder eller två funktionsord med små marginaler, och 20 par per kategori räcker inte för att låsa dem. Konsekvens för rapporten: en skillnad på under tio procentenheter i totalen mellan två svenska modeller är inte ett resultat, och testet behöver 50 par per kategori och tre frön per konfiguration innan kurvor ritas på det. Det som håller redan nu är mönstret: 0,3M under de övriga, pluralis på slumpnivå, genus/modal/tempus nära 100 %.

**Fröbruskontroll (01:04–01:39, `scripts/seed_check_sv.sh`, resultat i `checkpoints/seedcheck/`).** 0,3M och 1M tränades om på v1.1 med frö 1 och 2, samma budgetar som förut.

| Modell | Frö 0 (v1.1) | Frö 1 | Frö 2 | Bits/byte alla tre |
|---|---:|---:|---:|---:|
| pytte-0.3m-sv | 67,5 % | 74,5 % | 80,0 % | 0,538 / 0,535 / 0,539 |
| pytte-1m-sv | 76,0 % | 76,0 % | 77,0 % | 0,433 / 0,433 / 0,436 |

Vid 0,3M spänner tre frön 12,5 procentenheter på totalen vid identisk bits/byte, och enskilda kategorier hoppar 20–25 enheter (ordföljd 60–85 %, bisatsnegation 45–70 %, adjektivkongruens 65–90 %). Vid 1M är totalen stabil inom en enhet och bits/byte inom 0,003. Slutsats: grammatiktestet är pålitligt från 1M men inte vid 0,3M, där modellens val mellan två former avgörs av små marginaler som fröet rår på. Det ändrar inte Fynd 1 (0,3M skriver sammanhängande svenska, mätt på bits/byte och domare, som är stabila), men ett grammatikpåstående om 0,3M måste ges som ett intervall (68–80 %) och bygga på flera frön. Nedgången 1M → 76 % mellan v1.0 och v1.1 ligger utanför fröbruset vid 1M och är därmed troligen en verklig effekt av datarensningen, i storleksordningen tre enheter.

## 16. Fas 3d: SagoSvenska v2 startad, 2026-11-10 natt

Mathias godkände v2 01:05. Skrivare: Gemma 4 26B A4B (QAT 4-bit), temperatur 0,85, mål 60 000 sagor, start när GPU:n blir ledig efter fröbruskontrollen (ca 01:50), beräknad tid 35–40 timmar. Allt i `/Volumes/Modelltest/pytte/data/sagosvenska-v2/` (råsagor, `gen.log`, STOP/PAUSE-filer), kedjan är `scripts/chain_v2_gen.sh` → `data/gen_loop.sh` (batterivakt, omstart vid krasch) → `data/gen_stories.py --recipe v2`.

**Receptet v2** (`make_prompt_v2` i `data/gen_stories.py`; v1-receptet finns kvar som `--recipe v1`). Fynd 9 sa att mallen kommer från prompten, så v2 slumpar det som v1 höll fast:

| Ingrediens | v1 | v2 |
|---|---|---|
| Öppning | fri (82 % blev "Namn bor/är ...") | 14 öppningstyper: replik, ljud, fråga, väder, mitt i handling, känsla, tid på dygnet, "Det var en gång", räkning, sak/djur först, plats, rop, "inte", besök |
| Tempus | fri (nästan bara presens) | presens / preteritum 50/50 |
| Figurer | ett namn | en (45 %), två (35 %), tre barn (10 %), barn + vuxen (10 %) |
| Pluralis | inget | 40 % av sagorna får "flera {katter/bollar/...} i pluralis, så att böjningen syns" (60 pluralformer), mot Fynd 4c |
| Berättare | tredje person | 12 % av ensamsagorna i jag-form |
| Drag | ett av 12 | ett (60 %) eller två (40 %) av 29; slutdragen flyttade till egen lista |
| Slut | fritt | 11 sluttyper (lyckligt, alla somnar, roligt, fråga till lyssnaren, mening från början återkommer, överraskning, god natt, dela med sig, sorgligt men tryggt, återberättar, planerar morgondagen) |
| Meningar | "Korta meningar" | korta (70 %) / blandade (30 %) |
| Längd | 80–180 ord, jämnt | 60–220 ord, viktat mot 100–150, snitt 127 |
| Förbud | inget | börja inte med namn + bor/är/går; inte "Det är en fin dag", "Det är en solig dag", "Det är vinter i fjällen" |
| Ord/verb/adjektiv, plats | ett vardera | oförändrat |

Prompten blev ~280 token mot 178 (den delade systemdelen på 109 token cachas fortfarande), `--max-tokens 520` så att 220-ordssagor inte klipps (26B skriver 1,6–1,8 token per ord). Varje saga sparar alla ingredienser inklusive `recipe: "v2"`, så datakortet kan redovisa fördelningen exakt. Rensaren (`clean_stories.py`) läser bara `text`, `name` och `feature`, som finns i båda recepten.

**De första 540 sagorna (01:40–02:05, 1 550 sagor/h, ETA 38 h).** Jämfört på 500 slumpade per set med `scripts/pilot_stats.py`:

| Mått | v1 (12B, temp 0,6) | 26B A4B, v1-recept, temp 0,85 | **v2** |
|---|---:|---:|---:|
| Ord per saga | 110 | 118 | 117 |
| Unika ord (500 sagor) | 2 045 | 2 155 | 2 615 |
| Ord som täcker 90 % | 512 | 530 | 686 |
| 4-gram som återkommer i andra sagor | 29 % | 27 % | **16 %** |
| Börjar "Namn bor/är/går ..." | 82 % | 80 % | **4 %** |
| Sagor med dialog | 26 % | 41 % | 36 % |
| Flaggade | 12 % (specialtoken) | 0,4 % | 0,7 % (4 "unsafe", 0 specialtoken) |

Receptet gör det temperaturen inte gjorde: återanvändningen mellan sagor nästan halverad och ordförrådet 28 % större på samma antal sagor. Instruktionerna följs: alla 284 preteritum-prompter gav dåtid, 211 av 215 pluralprompter innehåller pluralordet, alla 26 jag-form-prompter innehåller "jag". Två skavanker syntes direkt och rättades 02:10 (generatorn startades om, löpande räkning fortsätter): ljud-öppningen kopierade promptens exempel ("Plask!" i 23 av 40 ljudsagor), så exemplen slumpas nu ur 20 ljud per prompt, och räkneexemplet "Tre små ankor" slumpas på samma sätt (antal × pluralord). Dessutom adderas antalet redan skrivna sagor till fröet vid omstart, så att en krasch inte spelar upp samma prompter igen. Första meningen är nu vanligast i 22 av 500 sagor mot 5 i v1, nästan helt "Plask!"; den siffran ska gå ner efter rättningen och kontrolleras när v2 är klar.

**Klart: 60 000 sagor 2026-11-12 11:20.** Genereringen tog 39,7 timmar effektiv tid fördelat på fyra körningar (01:40–09:04 den 10:e, 14:49–08:57, 18:10–18:56 och 21:59–11:20 den 11–12:e); de tre avbrotten var manuella pauser för att laptopen skulle med på resa, inte krascher. Farten låg på 1 510–1 557 sagor/h hela vägen och minnet stabilt på 13,5 GB. Omstarterna fungerade som tänkt: varje start fick nytt frö (frö + antal skrivna) och fortsatte räkningen. 389 sagor hamnade i review-filerna, alla på ordet "unsafe"-listan, ingen på specialtoken.

Tvätt med `data/clean_stories.py` (samma regler som v1.1) till `/Volumes/Modelltest/pytte/data/sagosvenska-v2/clean/`:

| | v1.1 | **v2** |
|---|---:|---:|
| Rårader (inkl. review) | 60 573 | 60 389 |
| Behållna | 60 426 | 60 223 |
| Bortfall | 138 unsafe, 7 citat, 2 engelska | 75 unsafe, 51 korta, 26 engelska, 5 citat, 2 ej svenska |
| Sagor med lagad specialtoken | 8 835 | **0** |
| Träning / val / test | 59 224 / 599 / 603 | 59 045 / 611 / 567 |
| Ord totalt | 6 673 991 | 6 975 141 |
| Ord per saga | 110 | 116 |
| Unika ord | 11 619 | **13 529** |
| Ord som täcker 90 % | 527 | **718** |

Jämförelse på 5 000 slumpade råsagor per set (`scripts/pilot_stats.py --n 5000`; siffrorna är större än i 500-tabellen ovan eftersom återanvändning växer med urvalet):

| Mått | v1 | **v2** |
|---|---:|---:|
| Ord per mening | 5,5 | 6,1 |
| Unika ord (5 000 sagor) | 4 552 | 5 668 |
| Ord som täcker 90 % | 530 | 714 |
| Olika öppningar (första 4 orden) | 39,5 % | 54,1 % |
| 4-gram som återkommer i andra sagor | 51,2 % | **36,9 %** |
| Sagor med dialog | 26 % | 34 % |
| Vanligaste första meningen, hela setet | "Det är en fin dag." 9 743 (16 %) | "Plask!" 1 236 (2,1 %) |
| Börjar "Namn bor/är/går ..." | 82 % | **4,4 %** |

Rättningen 02:10 gav effekt men inte helt: även med 20 slumpade ljud per prompt väljer modellen "Plask!" eller "Plopp!" i över hälften av ljudsagorna (1 236 + 1 147 av 4 333), så vattenljuden är favoriter oavsett vad som erbjuds. Det är ändå 2 % mot 16 % för v1:s vanligaste första mening. Öppningstyperna fördelar sig jämnt (4 183–4 407 per typ av 14), tempus exakt 50/50 (30 044 preteritum). Ingen tabell här är ett kvalitetsmått på språket; det mäts när familjen tränats om. Nästa steg: slå ihop v1.1 och v2 (13,6 miljoner ord, cirka 17 miljoner token) och träna om de sex svenska modellerna på den dubbla datan med samma tidsbudgetar, så att v1.1-raderna i avsnitt 13 får en direkt jämförelse.

## 17. Fas 3e: omträning på v1.1 + v2, 2026-11-12

Godkänt av Mathias 11:35 ("Ja, kör omträningen"). Syftet är en direkt jämförelse med avsnitt 13: samma sex modeller, samma tidsbudgetar (5/8/10/12/20/30 min), samma inlärningstakter och frö 0, men dubbelt så mycket och mer varierad data. Allt som skiljer är datan, så skillnaden i tabellen är datans effekt.

**Sammanslagningen** (`/Volumes/Modelltest/pytte/data/sagosvenska-all/clean/`, `stats.json` där): tvättade v1.1- och v2-splittarna lades ihop split för split, exakta dubbletter togs bort (0 hittades) och träningsdelen blandades med frö 0. Varje saga bär `recipe` v1 eller v2.

| Split | Sagor | varav v1.1 / v2 | Ord |
|---|---:|---:|---:|
| Träning | 118 269 | 59 224 / 59 045 | 13 378 588 |
| Validering | 1 210 | 599 / 611 | 136 602 |
| Test | 1 170 | 603 / 567 | 132 855 |

Valideringsförlusten mäts alltså på en blandning av v1- och v2-sagor, så bits/byte blir inte direkt jämförbar med avsnitt 13 (v2-sagor är mindre förutsägbara). Jämförbara är minimala par, domaren och nya 7-gram.

**Kedjan:** `scripts/chain_sv2.sh` under caffeinate, startad 11:40. `train_family.sh sv2` bygger `tok-sv2-2048` och `tok-sv2-4096` på den nya datan, tokeniserar, tränar och utvärderar varje modell till `checkpoints/pytte-<cfg>-sv2/`, sedan dömer `judge_family.py` 50 sagor per modell med Gemma 4 12B. Beräknad tid 2,5 timmar (85 min träning, resten tokenizer, utvärdering och domare). `sv2` är en datatagg, inte ett språk: skripten mappar den till de svenska minimala paren.

**Resultat (kedjan klar 14:00, träning 11:41–13:26, domare 13:26–14:00).** Rad sv = v1.1 (avsnitt 13), rad sv2 = samma modell tränad på v1.1 + v2. Bits/byte (bästa validering under träningen, som i avsnitt 13) är mätt på respektive egen valideringsmängd och därför inte jämförbar mellan raderna; se korstabellen nedan.

| Modell | data | token sedda | bpb (egen val) | min. par | per token | ord | slut | nya 7-gram | domare gr / sam / kre | språkfel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m | sv | 82M | 0,538 | 67,5 % | 69,5 % | 135 | 86 % | 50 % | 8,7 / 5,3 / 2,6 | 1,48 |
| pytte-0.3m | **sv2** | 87M | 0,654 | **82,5 %** | 82,5 % | 133 | 94 % | 65 % | 8,1 / 4,7 / 2,7 | 2,02 |
| pytte-1m | sv | 80M | 0,433 | 76,0 % | 77,0 % | 129 | 95 % | 44 % | 9,5 / 8,8 / 3,0 | 0,42 |
| pytte-1m | **sv2** | 85M | 0,525 | **91,5 %** | 89,5 % | 140 | 85 % | 61 % | 9,2 / 6,9 / 2,9 | 0,70 |
| pytte-2m | sv | 60M | 0,417 | 77,0 % | 77,5 % | 137 | 93 % | 45 % | 9,5 / 9,0 / 3,0 | 0,34 |
| pytte-2m | **sv2** | 57M | 0,506 | **89,0 %** | 88,0 % | 144 | 91 % | 60 % | 9,6 / 7,0 / 2,9 | 0,36 |
| pytte-3m | sv | 54M | 0,409 | 79,0 % | 80,5 % | 132 | 97 % | 45 % | 9,7 / 9,1 / 3,0 | 0,24 |
| pytte-3m | **sv2** | 56M | 0,491 | **90,0 %** | 90,0 % | 139 | 89 % | 59 % | 9,5 / 7,8 / 3,1 | 0,44 |
| pytte-7m | sv | 55M | 0,403 | 80,0 % | 80,0 % | 136 | 93 % | 44 % | 9,5 / 9,1 / 3,0 | 0,48 |
| pytte-7m | **sv2** | 59M | 0,472 | **95,0 %** | 94,0 % | 142 | 91 % | 61 % | 9,5 / 8,0 / 3,1 | 0,36 |
| pytte-15m | sv | 52M | 0,405 | 77,0 % | 80,0 % | 133 | 98 % | 44 % | 9,7 / 9,4 / 3,1 | 0,28 |
| pytte-15m | **sv2** | 53M | 0,461 | **90,0 %** | 89,0 % | 135 | 96 % | 58 % | 9,5 / 8,0 / 3,1 | 0,38 |

Minimala par per kategori, v1.1 → v1.1+v2 (20 par per kategori):

| Kategori | 0.3m | 1m | 2m | 3m | 7m | 15m |
|---|---:|---:|---:|---:|---:|---:|
| genus | 100 → 100 | 100 → 100 | 95 → 100 | 100 → 95 | 100 → 100 | 100 → 100 |
| bestämdhet | 65 → 80 | 90 → 90 | 85 → 90 | 100 → 90 | 80 → 85 | 90 → 90 |
| modal + infinitiv | 90 → 95 | 90 → 100 | 95 → 100 | 95 → 100 | 95 → 100 | 90 → 95 |
| preteritum | 75 → 95 | 85 → 100 | 85 → 95 | 90 → 100 | 95 → 100 | 90 → 100 |
| presens | 80 → 90 | 80 → 95 | 85 → 90 | 65 → 85 | 90 → 100 | 85 → 90 |
| ordföljd V2 | 70 → 80 | 75 → 95 | 75 → 90 | 80 → 100 | 70 → 100 | 75 → 95 |
| pronomen | 50 → 65 | 85 → 90 | 70 → 85 | 80 → 80 | 80 → 90 | 80 → 90 |
| bisatsnegation | 45 → 55 | 65 → 65 | 50 → 50 | 75 → 60 | 85 → 80 | 40 → 45 |
| adjektivböjning | 65 → 80 | 55 → 85 | 75 → 95 | 55 → 95 | 60 → 95 | 70 → 95 |
| **pluralis** | **35 → 85** | **35 → 95** | **55 → 95** | **50 → 95** | **45 → 100** | **50 → 100** |

Korskontroll av bits/byte (`data/tokens-cross/`, valideringsmängderna tokeniserade med respektive modells egen tokenizer, `pytte.eval --n-samples 1`):

| Modell | tränad på | bpb på v1-val | bpb på v2-val |
|---|---|---:|---:|
| pytte-1m | v1.1 | 0,436 | 1,080 |
| pytte-1m | v1.1 + v2 | 0,460 | 0,584 |
| pytte-3m | v1.1 | 0,410 | 1,063 |
| pytte-3m | v1.1 + v2 | 0,430 | 0,551 |
| pytte-15m | v1.1 | 0,407 | 1,068 |
| pytte-15m | v1.1 + v2 | 0,405 | 0,522 |

**Fynd 12: grammatiken i en pyttemodell är en dataegenskap, och den går att styra kategori för kategori.** Pluralis låg på slumpnivå i alla sex v1.1-modeller (35–55 %) och ligger nu på 85–100 %. Det enda som ändrats är att 40 % av v2-prompterna ber om flera saker i pluralis "så att böjningen syns". Adjektivböjning gick 55–75 → 80–95 %, ordföljd 70–80 → 80–100 %, preteritum 75–95 → 95–100 % (hälften av v2 är skriven i dåtid). Totalen steg 12–15 punkter för varje storlek (76 → 91,5 % vid 1M, 80 → 95 % vid 7M), långt över bruset från avsnitt 15 (±1 punkt från 1M). Bisatsnegation rörde sig inte (40–85 %, brus): den konstruktionen finns inte i något recept, så ingen data lärde ut den. Slutsatsen för rapporten: ett språkdrag som saknas i träningsdatan lärs inte in oavsett modellstorlek, och ett som finns i datan lärs in redan vid 0,3M parametrar. Det är ett starkare och mer användbart resultat än "svenska pluralis är svårt".

**Fynd 13: v1.1-modellerna hade lärt sig mallen, inte språket.** På v2-text kostar en v1.1-modell 1,06–1,08 bits/byte, mer än dubbelt mot sina 0,41–0,44 på egen text; modellerna såg v1 cirka tio gånger (80M token på 8M) och lärde sig den utantill. Modellerna tränade på v1.1 + v2 klarar v2-text på 0,52–0,58 och betalar bara 0,00–0,02 på v1-text. Samma tidsbudget, samma antal sedda token, men hälften så många varv över dubbelt så mycket data gav en modell som generaliserar. Nya 7-gram i egna sagor steg från 44–50 % till 58–65 %: mindre utantillrabbling. v1.1-bits/byte-siffrorna i avsnitt 13 (0,41) ska därför läsas som "på mallen", inte som ett mått på svenska.

**Fynd 14: priset är sammanhang.** Domaren ger sv2-modellerna 1,1–2,0 lägre sammanhang från 1M och uppåt (2M: 9,0 → 7,0; 3M: 9,1 → 7,8; 15M: 9,4 → 8,0) och 0,6 lägre vid 0,3M (5,3 → 4,7), medan grammatikbetyget är oförändrat (9,2–9,6) och språkfelen ligger kvar under 0,5 per saga från 2M. Fördelningen för 3M: v1.1 gav 31 av 50 sagor en tia, sv2 15; sv2 har nio fyror mot två. Domarens kommentarer på de lägsta pekar på samma sak varje gång: figurer byts ut mitt i ("Leon och Saga" blir "Juni" och "Lo"; "Kalle och Alfred" blir "Kalle och Yusuf"), hunden börjar tala. v2-receptet har mer än en figur i 55 % av sagorna (två eller tre barn i 45 %, barn och vuxen i 10 %) och fler namn att hålla reda på, och en modell med 64–256 dimensioner tappar vem som är vem. v1.1-sagorna med ett barn och en mall var lättare att hålla ihop. Så datan köpte grammatik och generalisering för sammanhang, vid samma tidsbudget. Två sätt att få tillbaka sammanhanget utan att ge upp vinsten: längre tidsbudget på den större datan (sv2-modellerna har sett den dubbla datan hälften så många gånger), eller en lättare blandning där v2 bidrar med grammatikdragen men enbarnssagor överväger. Det första är billigast att testa: pytte-1m i 16 min och pytte-3m i 24 min, cirka 45 min GPU. *(Rättat 2026-11-25: här stod 1,1–1,9 och "två eller tre barn i 55 %"; 2M tappar 2,0 och 55 % är sagorna med mer än en figur, se tabellen i avsnitt 16.)*

**Längre budget testad (godkänt 14:55, `scripts/sv2_long_budget.sh`, 15:02–16:02, resultat i `checkpoints/sv2long/`).** Samma data, tokenizer, inlärningstakt och frö som sv2, dubbel tid: pytte-1m 16 min (160M token, lika många varv över den dubbla datan som v1.1-modellen fick över sin) och pytte-3m 24 min (116M token).

| Modell | min | token | bpb egen val | bpb v1-val / v2-val | min. par | pluralis | slut | nya 7-gram | domare gr / sam / kre | språkfel | tior / ≤5 av 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-1m sv (v1.1) | 8 | 80M | 0,433 | 0,436 / 1,080 | 76,0 % | 35 % | 95 % | 44 % | 9,5 / 8,8 / 3,0 | 0,42 | 32 / 6 |
| pytte-1m sv2 | 8 | 85M | 0,525 | 0,460 / 0,584 | 91,5 % | 95 % | 85 % | 61 % | 9,2 / 6,9 / 2,9 | 0,70 | 12 / 17 |
| pytte-1m sv2 lång | 16 | 160M | 0,506 | 0,448 / 0,569 | 90,5 % | 100 % | 90 % | 62 % | 9,7 / 7,0 / 2,8 | 0,22 | 14 / 16 |
| pytte-3m sv (v1.1) | 12 | 54M | 0,409 | 0,410 / 1,063 | 79,0 % | 50 % | 97 % | 45 % | 9,7 / 9,1 / 3,0 | 0,24 | 31 / 2 |
| pytte-3m sv2 | 12 | 56M | 0,491 | 0,430 / 0,551 | 90,0 % | 95 % | 89 % | 59 % | 9,5 / 7,8 / 3,1 | 0,44 | 15 / 10 |
| pytte-3m sv2 lång | 24 | 116M | 0,471 | 0,412 / 0,530 | 92,0 % | 95 % | 92 % | 62 % | 9,5 / 7,9 / 3,0 | 0,46 | 18 / 10 |

**Fynd 15: sammanhanget kommer inte tillbaka med mer träning.** Dubbel tid gav 0,02 bättre bits/byte, 3 färre språkfel per tio sagor vid 1M och oförändrad grammatik, men sammanhanget flyttade sig 0,1 (6,9 → 7,0 och 7,8 → 7,9), inom bruset. Antalet sagor med sammanhang ≤5 är detsamma (16–17 av 50 vid 1M, 10 vid 3M). Förlusten i Fynd 14 beror alltså inte på att modellerna var undertränade utan på vad datan innehåller: sagor med flera figurer som modellen inte kan hålla isär vid den här storleken. Det avgör vägvalet: mer tid hjälper inte, en annan blandning av datan kan göra det.

Är det namnen? Domarens sammanhang per antal olika figurnamn i den bedömda sagan (namn ur generatorns namnlista, 50 sagor per modell):

| Modell | 1 namn | 2 namn | 3 namn |
|---|---:|---:|---:|
| pytte-1m sv | 8,8 (n=50) | – | – |
| pytte-1m sv2 | 7,3 (n=38) | 3,5 (n=2) | 5,8 (n=10) |
| pytte-3m sv | 9,1 (n=49) | 10,0 (n=1) | – |
| pytte-3m sv2 | 8,1 (n=39) | 8,0 (n=5) | 3,5 (n=2) |
| pytte-15m sv | 9,4 (n=49) | 10,0 (n=1) | – |
| pytte-15m sv2 | 8,3 (n=33) | 7,4 (n=13) | 9,0 (n=1) |

Delvis. Sagor med tre namn är riktigt dåliga (3,5–5,8) men få, och v1.1-modellerna skriver i praktiken aldrig fler än ett namn. Men även sv2-sagorna med ett enda namn ligger 1,0–1,5 under v1.1 (7,3 mot 8,8 vid 1M, 8,1 mot 9,1 vid 3M, 8,3 mot 9,4 vid 15M). Så namnen förklarar bara en del; resten är att v2-sagorna har mer innehåll att hålla ihop, med dialog, händelser, besökare och vändningar, och en pyttemodell har fler sätt att tappa tråden i en sådan saga än i "Namn går ut, ser X, blir glad". v1.1-modellens höga sammanhang är delvis en effekt av att mallen är enkel att fullfölja. Nästa experiment, om det ska göras: v1.1 + enbart v2-sagorna med ett barn (cast "en", cirka 27 000 sagor) för att se om pluralis- och tempusvinsten står kvar medan sammanhanget återhämtar sig; 1M och 3M med familjebudget, cirka en timme med domare.

**Blandningstestet (godkänt 2026-11-13, `scripts/sv3_mix_test.sh`, data i `data/sagosvenska-mix1/clean/`, resultat i `checkpoints/sv3mix/`).** All v1.1 plus v2-sagorna med `cast == "en"`: 85 764 träningssagor (59 224 + 26 540), 9,6 miljoner ord; val 853, test 865. Av v2:s 59 045 träningssagor är 26 540 enbarns, 20 677 tvåbarns, 6 000 trebarns och 5 828 barn + vuxen. Pluralis-, tempus- och jag-form-prompterna är oberoende av figurantalet, så de följer med i enbarnsdelen. Egen tokenizer (`tok-sv3-4096`), pytte-1m 8 min och pytte-3m 12 min, frö 0, domare 50 sagor.

**Resultat (07:57–08:38).** Tre blandningar, samma budget, samma frö. "sam. ≤1 namn" är domarens sammanhang på sagorna med högst ett figurnamn.

| Modell | min | bpb egen val | bpb v1-val / v2-val | min. par | plural / adj / ordföljd / pret | slut | nya 7-gram | domare gr / sam / kre | språkfel | tior / ≤5 av 50 | sam. ≤1 namn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-1m v1.1 | 8 | 0,433 | 0,436 / 1,080 | 76,0 % | 35 / 55 / 75 / 85 | 95 % | 44 % | 9,5 / 8,8 / 3,0 | 0,42 | 32 / 6 | 8,8 |
| pytte-1m **mix1** | 8 | 0,493 | 0,443 / 0,652 | **91,5 %** | 90 / 95 / 95 / 100 | 95 % | 53 % | 9,5 / **8,3** / 3,0 | 0,46 | 24 / 9 | 8,3 |
| pytte-1m v1.1 + v2 | 8 | 0,525 | 0,460 / 0,584 | 91,5 % | 95 / 85 / 95 / 100 | 85 % | 61 % | 9,2 / 6,9 / 2,9 | 0,70 | 12 / 17 | 7,3 |
| pytte-3m v1.1 | 12 | 0,409 | 0,410 / 1,063 | 79,0 % | 50 / 55 / 80 / 90 | 97 % | 45 % | 9,7 / 9,1 / 3,0 | 0,24 | 31 / 2 | 9,1 |
| pytte-3m **mix1** | 12 | 0,466 | 0,418 / 0,625 | **94,0 %** | 90 / 100 / 100 / 100 | 94 % | 54 % | 9,4 / **8,4** / 3,1 | 0,50 | 22 / 6 | 8,4 |
| pytte-3m v1.1 + v2 | 12 | 0,491 | 0,430 / 0,551 | 90,0 % | 95 / 95 / 100 / 100 | 89 % | 59 % | 9,5 / 7,8 / 3,1 | 0,44 | 15 / 10 | 7,9 |

**Fynd 16: enbarnsblandningen behåller grammatikvinsten och tar tillbaka det mesta av sammanhanget.** Minimala par 91,5 respektive 94,0 %, lika bra eller bättre än hela sammanslagningen, pluralis 90 %, adjektivböjning 95–100 %, ordföljd 95–100 %, preteritum 100 %. Sammanhanget gick från 6,9 till 8,3 vid 1M och från 7,8 till 8,4 vid 3M, alltså 0,5–0,7 under v1.1 i stället för 1,3–1,9; sagor med sammanhang ≤5 halverades (17 → 9 och 10 → 6) och andelen som avslutar själv är tillbaka på 94–95 %. Generaliseringen är kvar men mindre: 0,63–0,65 bits/byte på v2-text mot 1,06–1,08 för v1.1 och 0,55–0,58 för hela sammanslagningen, och på v1-text kostar den 0,01 mot v1.1. Resten av gapet i sammanhang (0,5–0,7) sitter kvar även i enbarnssagor (8,3 mot 8,8, 8,4 mot 9,1) och är priset för friare berättelser; det tar vi, eftersom alternativet är en modell som kan en mall. Beslut som föreslås: mix1 blir träningsdatan för den familj som släpps, alla sex storlekar tränas om på den med familjebudgetarna (cirka 2,5 timmar med domare), och v1.1- och v1.1+v2-familjerna behålls som jämförelse i rapporten (mall mot variation mot blandning är i sig ett resultat).

**Familjen på mix1 startad (godkänt 08:50, `scripts/chain_sv3.sh`, start 08:52).** Datataggen `sv3` i `train_family.sh` pekar på `sagosvenska-mix1`; tokenizer `tok-sv3-2048` byggs för 0,3M, `tok-sv3-4096` finns från blandningstestet. pytte-1m-sv3 och pytte-3m-sv3 från blandningstestet kopieras in som familjens modeller (samma konfiguration, data, budget och frö; tidsbudgeten gör ändå ingen körning bitexakt), så kedjan tränar 0,3M, 2M, 7M och 15M (65 min), utvärderar och dömer de fyra. Beräknat klart cirka 10:30.

**Klart 10:29 (träning 08:52–10:05, domare 10:05–10:29).** Hela familjen på mix1, `checkpoints/pytte-<storlek>-sv3/`. bpb från summary.json (bästa val), korskontroll med `tok-sv3-4096` (0,3M använder vocab 2048 och saknar korskontroll), domare Gemma 4 12B på 50 sagor (7M: 49 tolkade).

| Modell | min | token | bpb egen val | bpb v1-val / v2-val | min. par | pluralis | adj. | ordföljd | pret. | slut | nya 7-gram | domare gr / sam / kre | språkfel | tior / ≤5 av 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m sv3 | 5 | 87M | 0,619 | – | 82,0 % | 75 % | 85 % | 90 % | 90 % | 92 % | 61 % | 8,7 / 5,3 / 2,7 | 1,30 | 5 / 33 |
| pytte-1m sv3 | 8 | 83M | 0,493 | 0,443 / 0,652 | 91,5 % | 90 % | 95 % | 95 % | 100 % | 95 % | 53 % | 9,5 / 8,3 / 3,0 | 0,46 | 24 / 9 |
| pytte-2m sv3 | 10 | 54M | 0,478 | 0,429 / 0,641 | 88,5 % | 95 % | 85 % | 95 % | 95 % | 97 % | 54 % | 9,6 / 8,4 / 3,0 | 0,30 | 26 / 8 |
| pytte-3m sv3 | 12 | 54M | 0,466 | 0,418 / 0,625 | 94,0 % | 90 % | 100 % | 100 % | 100 % | 94 % | 54 % | 9,4 / 8,4 / 3,1 | 0,50 | 22 / 6 |
| pytte-7m sv3 | 20 | 55M | 0,451 | 0,405 / 0,608 | 94,5 % | 90 % | 90 % | 95 % | 100 % | 99 % | 54 % | 9,6 / 9,0 / 3,2 | 0,35 | 30 / 4 |
| pytte-15m sv3 | 30 | 47M | 0,445 | 0,400 / 0,603 | 93,0 % | 90 % | 100 % | 95 % | 100 % | 96 % | 53 % | 9,5 / 8,9 / 3,1 | 0,36 | 26 / 4 |

De tre familjerna sida vid sida, domarens sammanhang och minimala par (sv = v1.1, sv2 = v1.1 + v2, sv3 = mix1):

| Storlek | sammanhang sv / sv2 / sv3 | min. par sv / sv2 / sv3 | bpb på v2-val sv / sv2 / sv3 |
|---|---:|---:|---:|
| 0,3M | 5,3 / 4,7 / 5,3 | 67,5 / 82,5 / 82,0 % | – |
| 1M | 8,8 / 6,9 / 8,3 | 76,0 / 91,5 / 91,5 % | 1,080 / 0,584 / 0,652 |
| 2M | 9,0 / 7,0 / 8,4 | 77,0 / 89,0 / 88,5 % | – / – / 0,641 |
| 3M | 9,1 / 7,8 / 8,4 | 79,0 / 90,0 / 94,0 % | 1,063 / 0,551 / 0,625 |
| 7M | 9,1 / 8,0 / 9,0 | 80,0 / 95,0 / 94,5 % | – / – / 0,608 |
| 15M | 9,4 / 8,0 / 8,9 | 77,0 / 90,0 / 93,0 % | 1,068 / 0,522 / 0,603 |

**Fynd 17: blandningen håller över hela familjen, och från 7M är sammanhanget tillbaka.** Minimala par 88,5–94,5 % från 1M (mot 76–80 % för v1.1 och 89–95 % för hela sammanslagningen), pluralis 90–95 % i alla storlekar utom 0,3M (75 %), preteritum 95–100 %, ordföljd 95–100 %. Sammanhanget: 0,3M är tillbaka på v1.1-nivå (5,3), 1M–3M ligger 0,5–0,7 under v1.1 som i blandningstestet, och 7M och 15M ligger 0,1–0,5 under (9,0 mot 9,1; 8,9 mot 9,4), med 30 respektive 26 tior av 50 och bara 4 sagor på ≤5. De större modellerna har alltså kapacitet att hålla ihop de friare sagorna; de mindre betalar fortfarande lite. Generaliseringen ligger mellan de andra två som väntat: 0,60–0,65 bits/byte på v2-text, och på v1-text är 7M och 15M nu *bättre* än v1.1-modellerna (0,405 mot 0,403 och 0,400 mot 0,405, inom bruset). Språkfel 0,30–0,50 per saga från 1M, samma som v1.1. Antal figurnamn per saga är i praktiken ett (47–49 av 50), så namnförväxlingarna från Fynd 14 är borta. Bisatsnegation är fortfarande brus (55–85 %) och förblir rapportens exempel på ett drag som ingen data lär ut.

**Beslut: mix1 är släppdatan.** Familjen `pytte-*-sv3` är den som går vidare till SFT, run.c och demon. v1.1- och v1.1+v2-familjerna behålls som jämförelse i rapporten (mall mot variation mot blandning). 0,3M-modellen förblir den svaga länken (sammanhang 5,3, 33 av 50 sagor på ≤5) och ska presenteras som "skriver svenska meningar" snarare än "berättar sagor".


## 18. Fas 4a: chatt (SFT) på sv3-familjen, 2026-11-13

**Godkänt 10:35 ("SFT chat").** Målet är att "Berätta en saga om en älg" ger en saga om en älg, att "Vad händer sedan?" fortsätter den, och att modellen svarar kort och ärligt på hälsningar och frågor den inte kan. Allt detta med noll ny generering.

**Datan är gratis.** Varje saga i mix1 bär sina receptingredienser (namn, plats, substantiv, verb, adjektiv, pluralord, tempus, jag-form, drag, slut). `pytte/prepare_sft.py` skriver en kort svensk begäran ur en slumpad delmängd av dem (0–3 ingredienser; 14 öppningsfraser som "Berätta en saga", "Kan du hitta på en saga", "Jag vill höra en godnattsaga"; 15 % gemener, 15 % utan skiljetecken, 10 % "tack") och sagan blir svaret. Ordformerna tas ur korpusen i stället för att gissas: artikel per substantiv (en/ett efter vilken som är vanligast i sagorna, inget för massord som snö och mjölk) och presensform per verb (hoppar/bygger/springer efter frekvens, ett undantag: "tappa", där adjektivet tapper vann räkningen). Adjektiv sätts bara attributivt framför en-ord ("en blyg älg"); vid ett-ord hamnar det på barnet ("där Elsa är blyg"), så ingen böjning behöver gissas.

Samtalstyper (träning 83 507 samtal, 14,3 M token varav 12,0 M i förlusten; validering 828 samtal ur val-sagorna):

| Typ | Antal | Innehåll |
|---|---:|---|
| enkel | 48 851 | begäran → hel saga |
| fortsättning | 28 327 | begäran → första halvan (klipp vid meningsgräns 35–65 % in), "Vad händer sedan?"/"Fortsätt."/"Och sen då?" → resten |
| en till | 4 293 | hel saga, "En till!" → nästa saga |
| hälsning, vem är du, vad kan du, tack, hej då | 1 425 | korta fasta svar med variation ("Hej! Jag heter Pytte. Vill du höra en saga?") |
| utanför ämnet | 611 | 34 frågor av typen "Vad är huvudstaden i Frankrike?", "Skriv ett program" → "Det vet jag inte. Jag är en pytteliten modell och kan bara berätta sagor. Vill du höra en saga?" |

**Format.** Ren text, inga nya specialtoken (tokeniseraren har bara `<|endoftext|>`, och formatet ska fungera i run.c och i webbläsaren utan vidare): `<|endoftext|>Du: Berätta en saga om en älg.\nPytte: <saga><|endoftext|>`. Varje Pytte-tur slutar med `<|endoftext|>`, en följdfråga läggs efter den och modellen fortsätter. "Du:" och "Pytte:" kostar 2 + 5 token per tur ("Användare:" hade kostat 6). Förlusten räknas bara på Pytte-token (mask i `train.mask.bin`, `pytte.train` läser den om den finns), modellen startas från `pytte-<storlek>-sv3` (`--init`), lägre inlärningstakt (5e-4) och en budget på ungefär en epok. `pytte/chat.py` sköter formatet, strömmar svaret och stoppar på `<|endoftext|>`.

**Mätning (`scripts/eval_sft.py`).** 100 nya begäranden byggda ur val-sagornas recept (andra sagor än träningen): andel begärda ingredienser som finns i svaret (namn exakt; substantiv som ordstam; adjektiv och verb med böjningsformer; plats som sista ordet; pluralord exakt; jag-form som "jag"; dåtid som fler preteritum- än presensverb), andel svar med alla ingredienser, egen avslutning och längd. 30 följdfrågor: fortsätter svaret med ny text (4-gram-överlapp mot första delen) och behåller namnet? 24 hälsningar/vem-är-du/utanför-ämnet: kort svar (≤40 ord) som erbjuder en saga? Därtill minimala par (grammatiken ska inte försämras) och domaren på de 50 första sagosvaren.

**Pilot (`scripts/sft_pilot.sh`, start 10:45): pytte-3m-sv3 → pytte-3m-sv3-chat, 4 min, lr 5e-4.**

**Pilotresultat (10:45–10:58).** 2 267 steg, 18,6 M token (1,3 epoker), val-förlust på Pytte-token 1,449 → 1,395 och fortfarande sjunkande.

| Mått | pytte-3m-sv3-chat, 4 min |
|---|---:|
| begärd ingrediens finns i svaret | 59 % (namn 93 %, plats 79 %, dåtid 4/5, substantiv 44 %, verb 18 %, adjektiv 15 %, pluralord 0/2) |
| alla begärda ingredienser | 43 % |
| avslutar själv / ord | 99 % / 101 |
| följdfråga efter hel saga: ny text ≥10 ord / behåller namnet / ord | 67 % / 63 % / 24 |
| hälsning / vem är du / utanför ämnet: kort svar | 7/8 / 6/8 / 4/8 |
| minimala par | 92,5 % (bas 94,0 %) |
| domare gr / sam / kre, språkfel | 9,6 / 8,8 / 2,8, 0,30 (bas 9,4 / 8,4 / 3,1, 0,50) |

Tre saker syns direkt. (1) Formatet fungerar och sagan tar ingen skada: grammatiken ligger kvar, domarens sammanhang gick upp (8,4 → 8,8, sagor med en given huvudperson är lättare att hålla ihop) och 99 % avslutar själva. (2) Modellen följer det som står först och ofta (namnet 93 %) men tappar saker som ska dyka upp senare i sagan (substantiv 44 %, adjektiv 15 %): det är underträning, inte oförmåga, eftersom val-förlusten inte hunnit plana ut. Startförlusten (1,45) låg dessutom över basmodellens 1,33 på ren sagotext, så inlärningstakten 5e-4 knuffade modellen ur sitt minimum och fyra minuter räckte inte för att komma tillbaka. (3) "Vad händer sedan?" efter en *hel* saga gav en mening och stopp ("Det är en fin kväll."), för träningen hade bara följdfrågor efter en halv saga. Åtgärd i datan: 10 % "avsnitt", där följdfrågan efter en hel saga besvaras med en ny saga om samma barn (1 400 sagor per namn gör det gratis). Småpratet är halvvägs: rätt svar på de flesta hälsningar, men "Vem är du?" och frågor utanför ämnet får ibland en saga; även det är en epok-fråga.

**Svep på 3M (`scripts/sft_sweep.sh`, start 11:14, `checkpoints/sftsweep/`).** Samma data med avsnitten tillagda (83 544 samtal, 15,0 M token, 12,7 M i förlusten): (a) lr 1e-4 i 8 min, (b) lr 3e-4 i 8 min, (c) lr 5e-4 i 12 min. Utan domare; den bästa döms efteråt.

**Svep 1, resultat (11:14–11:45).** Samma 100 begäranden, samma frö.

| Inställning | steg / token | val-förlust | ingrediens i svaret | alla | namn | plats | substantiv | verb | adjektiv | följdfråga: ny text / namn kvar / ord | småprat kort: hälsn. / vem / utanför | min. par |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot 5e-4, 4 min (utan avsnitt) | 2 267 / 19M | 1,395 | 59 % | 43 % | 93 % | 79 % | 44 % | 18 % | 15 % | 67 % / 63 % / 24 | 7/8 / 6/8 / 4/8 | 92,5 % |
| a) 1e-4, 8 min | 4 356 / 36M | 1,380 | 60 % | 43 % | 90 % | 92 % | 44 % | 24 % | 15 % | 100 % / 90 % / 92 | 3/8 / 2/8 / 3/8 | 92,5 % |
| b) 3e-4, 8 min | 4 282 / 35M | 1,374 | 60 % | 42 % | 94 % | 92 % | 37 % | 24 % | 15 % | 97 % / 93 % / 84 | 6/8 / 3/8 / 4/8 | 92,5 % |
| c) 5e-4, 12 min | 6 659 / 55M | 1,372 | 66 % | 46 % | 99 % | 100 % | 45 % | 35 % | 15 % | 100 % / 90 % / 83 | 6/8 / 5/8 / 4/8 | 93,5 % |

Avsnittsdatan gjorde sitt: följdfrågan efter en hel saga ger nu en ny saga om samma barn i 90 % av fallen (83–92 ord, 1–3 % 4-gram-överlapp med den första). Mer tid hjälper det som redan fungerar (namn 99 %, plats 100 %, verb 35 %), men substantivet står still på 37–45 % och adjektivet på 15 % oavsett inlärningstakt och tid. Det är inte datan: i träningssagorna finns substantivet i 98 %, adjektivet i 96 % och verbet i 95 % av fallen (mätt med samma kontroll). Mönstret i c: substantivet följs i 54 % när det är enda ingrediensen och 40 % när det är en av två eller tre; när det följs kommer det tidigt (median efter en fjärdedel av sagan); en-ord 47 %, ett-ord 38 %. Tolkning: namnet lyckas för att förträningen redan lärt modellen att kopiera ett namn genom hela sagan, platsen för att den står i första meningen, medan "nämn en älg någon gång" är en ny förmåga som kräver att modellen hämtar ett ord ur begäran mitt i berättelsen, och med substantivet i hälften av begärandena och en förekomst per saga är signalen svag. Två spakar kvar: tätare signal i datan och mer tid. Svep 2 (`scripts/sft_sweep2.sh`, köat efter svep 1, start 11:45): "tät" data (`--dense`: 2–3 ingredienser i 72 % av begärandena, substantivet i 76 % i stället för cirka 50 %), (d) 3e-4 i 8 min som jämförelse med b, (e) 3e-4 i 20 min.

**Svep 2, resultat (11:45–12:14).**

| Inställning | steg / token | val-förlust | ingrediens i svaret | alla | namn | plats | substantiv | verb | adjektiv | följdfråga: ny text / namn kvar / ord | småprat kort: hälsn. / vem / utanför | min. par |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b) gles data, 3e-4, 8 min | 4 282 / 35M | 1,374 | 60 % | 42 % | 94 % | 92 % | 37 % | 24 % | 15 % | 97 % / 93 % / 84 | 6/8 / 3/8 / 4/8 | 92,5 % |
| d) tät data, 3e-4, 8 min | 4 250 / 35M | 1,378 | 65 % | 45 % | 96 % | 92 % | 48 % | 35 % | 19 % | 100 % / 100 % / 103 | 8/8 / 3/8 / 2/8 | 94,5 % |
| e) tät data, 3e-4, 20 min | 11 427 / 94M | 1,351 | **74 %** | **58 %** | 99 % | 96 % | **65 %** | 41 % | 31 % | 97 % / 97 % / 95 | 6/8 / 6/8 / 4/8 | 91,5 % |

Båda spakarna verkar, och tiden är den stora: tät data gav +11 punkter på substantivet vid samma budget (37 → 48 %), och 20 minuter gav ytterligare +17 (48 → 65 %) med val-förlusten fortfarande sjunkande (1,378 → 1,351). Adjektivet dubblades (15 → 31 %). Sagans kvalitet står kvar: minimala par 91,5–94,5 % (bas 94,0; svängningen är testets brus från avsnitt 15, men adjektivkongruens 0,85 vid 20 min är värt att hålla ögonen på i familjen). Småprat är fortfarande ojämnt vid 3M (hälsningar rätt, "vem är du" och utanför-ämnet hälften rätt).

**Beslut och start 12:15: chattfamiljen (`scripts/chain_sft_family.sh -dense 3e-4 2.5`, logg `checkpoints/sft_family.log`).** Tät data, lr 3e-4, 2,5 × förträningsbudgeten per storlek (0,3M 12,5 min, 1M 20, 2M 25, 3M 30, 7M 50, 15M 75; summa 3,5 h) plus mätning och domare; beräknat klart cirka 16:30. 3M-punkten vid 30 min förlänger kurvan 8 → 20 → 30 min, och frågan för de större modellerna är om de följer substantivet bättre vid samma antal epoker (hypotesen från Fynd 17: mer kapacitet, bättre bindning). Pilotens checkpoint ligger kvar som `checkpoints/sftsweep/pytte-3m-sv3-chat-pilot`, svepet som `-a` … `-e`.

**Chattfamiljen, resultat (12:15–21:30, med två pauser för möten).** Tät data (83 474 samtal, 15,2 M token, 12,7 M i förlusten), lr 3e-4, cosinus till 2e-5, 100 begäranden med minst en ingrediens, 30 följdfrågor, 24 småpratsrepliker, 200 minimala par, domare på 50 svar. 3M stoppades vid 25,1 av 30 min för ett möte (inlärningstakten var redan nere på 4e-5, så kurvan var i stort sett klar); 15M stoppades vid 27,1 av 75 min och kördes om från grunden i full tid, den avbrutna ligger kvar som `pytte-15m-sv3-chat-27min` och står med i tabellen som en 3-epokspunkt.

| Modell | min | steg / token (epoker) | val-förlust | ingrediens i svaret | alla | namn | plats | substantiv | verb | adjektiv | följdfråga: ny text / namn kvar / ord | småprat kort: hälsn. / vem / utanför | min. par (bas sv3) | domare gr / sam / kre / fel (bas sam) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m-chat | 12,5 | 28 331 / 232M (15) | 1,699 | 51 % | 31 % | 90 % | 67 % | 29 % | 24 % | 8 % | 100 % / 87 % / 67 | 4/8 / 0/8 / 2/8 | 81,5 % (82,0) | 8,8 / 6,0 / 2,8 / 1,34 (5,3) |
| pytte-1m-chat | 20 | 25 184 / 206M (14) | 1,433 | 70 % | 54 % | 100 % | 92 % | 47 % | 53 % | 35 % | 100 % / 97 % / 94 | 8/8 / 7/8 / 5/8 | 93,0 % (91,5) | 9,7 / 8,3 / 2,9 / 0,22 (8,3) |
| pytte-2m-chat | 25 | 19 007 / 156M (10) | 1,376 | 74 % | 60 % | 100 % | 92 % | 66 % | 41 % | 19 % | 100 % / 100 % / 101 | 5/8 / 5/8 / 6/8 | 86,5 % (88,5) | 9,7 / 8,1 / 3,0 / 0,28 (8,4) |
| pytte-3m-chat | 25 av 30 | 14 750 / 121M (8) | 1,348 | 76 % | 57 % | 100 % | 100 % | 56 % | 71 % | 35 % | 100 % / 100 % / 98 | 6/8 / 6/8 / 4/8 | 91,5 % (94,0) | 9,7 / 8,7 / 2,9 / 0,22 (8,4) |
| pytte-7m-chat | 50 | 18 006 / 148M (10) | 1,327 | **79 %** | **66 %** | 99 % | 92 % | **74 %** | 53 % | **42 %** | 100 % / 100 % / 97 | 8/8 / 7/8 / 7/8 | 91,5 % (94,5) | 9,6 / **9,3** / 3,0 / 0,32 (9,0) |
| pytte-15m-chat, 27 av 75 min | 27 | 5 750 / 47M (3) | 1,358 | 69 % | 56 % | 100 % | 92 % | 56 % | 29 % | 23 % | 100 % / 97 % / 90 | 8/8 / 7/8 / 8/8 | 88,5 % (93,0) | 9,7 / 9,1 / 3,2 / 0,30 (8,9) |
| pytte-15m-chat | 75 | 16 298 / 134M (9) | 1,358 (steg 5 250) | 72 % | 56 % | 100 % | 96 % | 50 % | 41 % | **46 %** | 100 % / 97 % / 91 | 8/8 / 8/8 / 7/8 | 88,5 % (93,0) | 9,7 / 8,9 / 3,0 / 0,32 (8,9) |

**Fynd 18: instruktionsföljning växer med kapaciteten, och sagan tar ingen skada.** (1) Vid lika många epoker följer den större modellen fler ingredienser: 7M vid 10 epoker träffar 79 % mot 2M:s 74 % vid 10 epoker, substantivet 74 mot 66 % och adjektivet 42 mot 19 %, så hypotesen från Fynd 17 (mer kapacitet, bättre bindning mellan begäran och saga) håller även här. 0,3M når 51 % efter 15 epoker och 29 % på substantivet; den klarar namnet (90 %) men inte att hämta ett ord mitt i sagan, och det är kapacitet, inte tid. (2) Epokerna är den andra spaken: 15M efter 3 epoker (69 %) ligger under 1M efter 14 (70 %) trots lägre val-förlust, så val-förlusten mäter sagotexten, inte följsamheten; följsamheten kommer senare i träningen än språket. (3) Svårighetsordningen är stabil i hela familjen: namn (90–100 %) och plats (67–100 %) som står i sagans första mening, sedan substantivet (29–74 %), verbet (24–71 %) och sist adjektivet (8–42 %), som ska in före ett substantiv modellen själv valt. (4) Följdfrågorna fungerar från 1M: ny text i 100 %, samma barn i 97–100 %, 1–2 % 4-gram-överlapp med första delen. Småpratet kräver 1M (0,3M svarar med saga på nästan allt) och sitter bra från 7M (22 av 24 korta, rätt svar). (5) Sagans kvalitet står kvar eller blir bättre: domarens grammatik 9,6–9,7 i hela familjen (bas 9,4–9,6), sammanhanget upp vid 0,3M (5,3 → 6,0) och från 3M (+0,2–0,3), oförändrat vid 1M och −0,3 vid 2M (ett givet barn och en given plats gör sagan lättare att hålla ihop), språkfel 0,2–0,3 per saga (bas 0,3–0,5); minimala par ligger 0–3 punkter under basen, inom testets brus från avsnitt 15, men adjektivkongruensen (0,75–0,95) och pluralen (0,65–0,90) är de kategorier som rör sig, vilket är rimligt när träningen nu belönar att ett begärt adjektiv och en begärd plural sätts in. (6) **15M överanpassar chattdatan.** Dess val-förlust bottnar vid steg 5 250 (24 min, 43 M token, 3 epoker) och stiger sedan till 1,42–1,44 vid 75 min, medan 7M:s sjunker till steg 14 250 och 3M:s hela vägen. Bästa-val-checkpointen är därför den tidiga, och den fulla 75-minuterskörningen (72 %, substantiv 50 %, adjektiv 46 %) hamnar bara 3 punkter över den avbrutna 27-minuterskörningen (69 %) med samma val-förlust 1,358. 15 M parametrar lär sig 12,7 M förlusttoken utantill på tre epoker; 7M, med hälften så många parametrar, gör det inte på tio. Så 7M är den bästa chattmodellen i dag, och 15M-chat behöver mer *variation* i datan, inte mer tid: varje saga kan ge flera olika begäranden (olika ingrediensurval, olika delningspunkter), vilket är gratis och skulle tre–fyrdubbla datan utan en enda ny saga. Det är nästa steg om 15M-chat ska släppas som bästa modell; annars släpps den som den är med noten ovan. **Beslut:** chattfamiljen släpps vid sidan av basmodellerna som `pytte-<storlek>-chat`; demon kör chattmodellen, och 7M är referensen för "följer begäran". Tabellskriptet: `scripts/chat_family_table.py`.

**Test av tätare data (x3), 2026-11-14 10:58–13:18 (`scripts/chain_sft_x3.sh`, `prepare_sft.py --per-story 3`).** Tre pass över samma 85 764 sagor med nya slumpdragningar (ingrediensurval, formulering, delningspunkt, samtalstyp) gav 244 562 unika samtal, 45,2 M token, 37,9 M i förlusten, utan en enda ny saga. 15M och 7M tränades om på den datan med samma budgetar (75 och 50 min); de ursprungliga ligger kvar som jämförelse.

| Modell | data | steg / token (epoker) | bästa val (steg) / sista | ingrediens i svaret | alla | substantiv | verb | adjektiv | följdfråga namn kvar | småprat kort | min. par | domare gr / sam / fel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-7m-chat | x1, 15 M tok | 18 006 / 148M (10) | 1,327 (14 250) / 1,350 | **79 %** | **66 %** | **74 %** | 53 % | 42 % | 100 % | 22/24 | 91,5 % | 9,6 / 9,3 / 0,32 |
| pytte-7m-chat-x3 | x3, 45 M tok | 18 143 / 149M (3,3) | 1,326 (18 000) / 1,345 | 72 % | 56 % | 53 % | 53 % | 38 % | 100 % | 0/24 | 92,5 % | 9,6 / 9,1 / 0,36 |
| pytte-15m-chat | x1, 15 M tok | 16 298 / 134M (9) | 1,358 (5 250) / 1,419 | 72 % | 56 % | 50 % | 41 % | 46 % | 97 % | 23/24 | 88,5 % | 9,7 / 8,9 / 0,32 |
| pytte-15m-chat-x3 | x3, 45 M tok | 15 117 / 124M (2,7) | 1,349 (8 750) / 1,410 | 75 % | 58 % | 56 % | 53 % | **54 %** | 93 % | 1/24 | 91,5 % | 9,8 / 9,1 / 0,18 |

**Fynd 19: vid fast tidsbudget är upprepning, inte variation, det som lär modellen följa begäran.** 7M tappade 7 punkter på x3-datan (79 → 72 %, substantiv 74 → 53 %) trots lika många steg och lägre val-förlust: med tre gånger så många olika begäranden hann den bara 3,3 epoker i stället för 10, och det är de upprepade passen över samma saga–begäran-par som bygger kopplingen "ordet i begäran ska in i sagan". Val-förlusten säger inget om det (x3-modellerna har lägst val-förlust och sämst följsamhet), vilket bekräftar punkt 2 i Fynd 18. För 15M sköt variationen upp överanpassningen (val-minimum vid steg 5 250 → 8 750) och gav +3 punkter (72 → 75 %, adjektiv 46 → 54 %), men kurvan vänder fortfarande uppåt, så 15 M parametrar överanpassar även 38 M förlusttoken, och 15M-chat förblir sämre än 7M-chat på att följa begäran. Adjektivet är det enda som vann på variationen (7M 42 → 38 är brus, 15M 46 → 54), rimligt eftersom adjektivet har flest möjliga kombinationer per saga. **Bugg i testet:** min dedupe av exakta dubbletter tog bort nästan allt småprat (fasta fråge- och svarslistor ger få unika kombinationer: 279 rader i x3 mot 2 035 i x1), så x3-modellerna svarar med en saga på "Hej" (0 av 24 korta). Rättat i `prepare_sft.py` (dedupe bara på sagosamtal); följsamhetssiffrorna påverkas inte (småpratet är 2 % av datan). **Beslut:** x1-modellerna är chattfamiljen (7M referens, 15M släpps med noten om överanpassning), x3-modellerna ligger kvar som `pytte-{7m,15m}-sv3-chat-x3`. Om 15M-chat ska bli bäst krävs regularisering (dropout, viktavklingning, lägre inlärningstakt eller en kortare budget med fler pass), inte mer data av samma slag; det är inte prioriterat före demo och rapport.


## 19. Fas 4b: export och run.c, 2026-11-24

**Mål.** Köra Pytte utan Python: en fil som innehåller modell och tokeniserare, och ett C-program utan beroenden som läser den. Samma fil ska webbdemon ladda, så formatet är gjort för att vara enkelt att läsa i JavaScript.

**Formatet (`pytte/export.py`, version 1).** Magic `PYTT`, header med dimensioner (dim, lager, huvuden, kv-huvuden, ff, vokabulär, kontext, kvantisering, knutna inbäddningar, RoPE-bas, norm-epsilon), sedan tokeniseraren (varje tokens råa byte, `<|endoftext|>`-id, sammanslagningsreglerna som id-tripplar i rangordning) och sedan vikterna i framåtordning: inbäddning, per lager attn-norm, wq, wk, wv, wo, mlp-norm, w1, w3, w2, och slutnormen. Matriser lagras som int8 per rad (symmetrisk skala = max|w|/127, skalorna som f32 efter matrisen), normvektorer som f32. `--f32` ger samma fil utan kvantisering, för kontroll. Tokeniseraren är HF:s byte-nivå-BPE: tokensträngarna avkodas från GPT-2:s byte→unicode-tabell till råa byte vid exporten, så C-koden slipper tabellen.

**`run.c` (cirka 450 rader).** Hela framåtpasset i ordning: GPT-2:s förtokenisering (regexen `'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` som en handskriven skanner över UTF-8-kodpunkter, med bokstavsklassning för latinska, grekiska och kyrilliska intervall), exakt BPE med sammanslagningstabellen (hashtabell par → rang och resultat, inte "lägsta rang vars sträng matchar"), inbäddning, RMSNorm, attention med RoPE (MLX:s icke-traditionella variant, paren (i, i+d/2)), grouped-query attention, KV-cache, SwiGLU, knutet LM-huvud, sampling med temperatur, top-k och top-p (samma regel som `pytte.sample`). Lägen: fortsätt en text, saga från ingenting, `-chat` (formatet från avsnitt 18 med historik, äldsta turerna faller bort när kontexten är full, `/ny` börjar om, stoppar på `<|endoftext|>` och på `\nDu:` som säkerhetsnät), `-tok` som visar token, `-score` som ger en texts logaritmiska sannolikhet (så grammatiktestet kan köras genom C-koden). `cc -O3 -o run run.c -lm`.

**Kontroll (`scripts/check_run.py`, `scripts/pairs_run.py`, `scripts/q8_eval.py`).** Tre lager: (1) tokeniseraren ger identiska id på 300 valideringssagor plus kantfall (dubbla mellanslag, radbrytningar, engelska apostrofer, tankstreck) för 7M, och 50 + kantfall för 0,3M och 15M: 100 %. (2) f32-filen ger exakt samma giriga fortsättning som Python från åtta promptar (80 token) för 0,3M (GQA, vokabulär 2048), 7M och 15M: 8/8, 8/8, 8/8, och de minimala paren genom `-score` ger exakt Pythons 91,5 % för 7M-chat. (3) int8-filen: viktfelet är 0,8 % i medel, och 5/8 giriga fortsättningar är identiska i 60–80 token (girig avkodning förstorar varje avrundning till en annan text, därför mäts kvaliteten med testerna i stället).

**Hastighet.** Den första int8-matrisprodukten (int8-vikt × f32-aktivering, flyttalsackumulering) vektoriserades inte av clang och gav 210 tok/s för 7M oavsett `-O3`/`-ffast-math`/`-march=native`. Att kvantisera aktiveringen till int8 (en skala per vektor) och ackumulera int8 × int8 i int32 vektoriseras och gav 4 000 tok/s: 14–19 gånger snabbare, med en kärna.

| Modell | .bin | tok/s (en kärna, M2 Pro) | min. par genom run.c, int8 (Python, f32) | chat-.bin min. par (Python) |
|---|---:|---:|---:|---:|
| pytte-0.3m | 0,38 MB | 38 000 | 81,5 % (82,0) | 81,5 % (81,5) |
| pytte-1m | 1,44 MB | 15 500 | 90,5 % (91,5) | 92,0 % (93,0) |
| pytte-2m | 2,37 MB | 10 100 | 89,5 % (88,5) | 86,0 % (86,5) |
| pytte-3m | 3,59 MB | 6 800 | 94,0 % (94,0) | 91,5 % (91,5) |
| pytte-7m | 7,33 MB | 3 900 | 94,5 % (94,5) | 90,5 % (91,5) |
| pytte-15m | 15,97 MB | 1 900 | 92,5 % (93,0) | 88,5 % (88,5) |

**Fynd 20: int8 med int8-aktiveringar kostar ingenting som syns.** Vikter och aktiveringar i int8 (det C-koden faktiskt räknar med) ger minimala par inom ±1 punkt från f32-modellerna på alla tolv filerna, och det fullständiga chattestet på 7M-chat med int8-vikter (`q8_eval.py`, f32-aktiveringar) ger 76,2 % ingredienser mot 78,6 % och samma minimala par (92,0 mot 91,5): allt inom testernas brus. Filerna är 4 gånger mindre än safetensors (7,3 mot 28,7 MB för 7M) och 15M-chat, den största, är 16 MB. Hela familjen ligger i `/Volumes/Modelltest/pytte/bin/` (tolv filer, 0,38–16 MB). Make-mål: `make run`, `make export CONFIG=pytte-7m LANG_=sv3-chat`.

**Vad som inte gjordes.** Ingen parallellisering (en kärna räcker: 7M-chat skriver 25 gånger snabbare än någon läser), ingen int4 (inte nödvändigt vid 16 MB), ingen ESP32/Game Boy-portning (kvar som "valfritt" i fas 4). Fejkkvantiseringskatalogen `checkpoints/pytte-7m-sv3-chat-q8/` ligger kvar från mätningen.

## 20. Fas 4c: webbdemon, 2026-11-24

**Mål.** Demon från avsnitt 4: allt körs i besökarens webbläsare, ingen server, inget skickas någonstans. Statiska filer som kan ligga på GitHub Pages eller en Hugging Face Space.

**Bygge (`web/`, fem filer plus modellerna).** `pytte.js` är run.c översatt till JavaScript, steg för steg: samma filformat (DataView; viktdelen kopieras till en egen buffert så att alla f32-vyer blir 4-bytejusterade), samma förtokenisering (här som GPT-2:s regex direkt, `\p{L}` med `u`-flaggan, i stället för C-skannern), samma exakta BPE via sammanslagningstabellen, samma framåtpass med int8-vikter och int8-aktiveringar (heltalsprodukterna är exakta i JavaScripts dubbelprecision), samma sampling och samma xorshift-slumpgenerator med samma frösättning (BigInt). `worker.js` kör modellen i en Web Worker så att sidan svarar medan den skriver, strömmar token ett i taget, håller tillbaka text som kan vara början på stoppsträngen `\nDu:` i chatten och tar emot avbrott. `app.js` är gränssnittet, `index.html` och `style.css` sidan (varumärkets färger från `brand/`, mobilanpassad). Modellfilerna ligger som kopior i `web/models/` (tolv filer, 59 MB; symbollänk till den externa disken gick inte, se nedan). Lokal server: `python3 -m http.server 8765 --directory web` (`.claude/launch.json`).

**Tre flikar.** (1) *Sagan växer*: en början, ett frö, ett reglage med sex storlekar; varje drag genererar om samma saga med nästa modell, så man ser språket uppstå mellan 0,3 M och 7 M. (2) *Prata med Pytte*: chattformatet från avsnitt 18 med historik, uppföljningar och storleksval (7M-chat som standard). (3) *Under huven*: texten uppdelad i token med id (bitar av ord, mellanslag som ␣) och de tolv mest sannolika nästa token som staplar, uppdaterat medan man skriver. Uppmärksamhetsmönstren från avsnitt 4 är inte med i första versionen.

**Kontroll.** Samma frö och samma början ger byte för byte samma saga i webbläsaren som i `run.c` (2M, frö 624419, 81 token), så JavaScript-implementationen är verifierad mot C-implementationen, som i sin tur är verifierad mot Python (avsnitt 19). *(Rättat 2026-11-25: det här var en enda saga på 81 token, och slutsatsen höll inte. Se avsnitt 24: av 70 sagor på upp till 250 token var 20 identiska hela vägen.)* Hastighet i Chromium på M2 Pro, en tråd ren JavaScript utan WebAssembly eller WebGPU: 0,3M 640 token/s, 2M 220, 7M-chat 43, 15M 55 (den sista mätningen på 60 token, jit-uppvärmning ingår). Det är 25–60 gånger långsammare än run.c men fortfarande snabbare än någon läser, så WebAssembly får vänta tills det behövs.

**Problem på vägen.** Serverprocessen som appen startade hängde på allt under `web/models/` när den var en symbollänk till `/Volumes/Modelltest`: processen saknar rättighet till flyttbara volymer och förfrågan blockerades utan fel. Riktiga kopior i `web/models/` löste det, och det är ändå så demon ska distribueras.

**Kvar till lansering.** Länkar i sidfoten (kod, modeller, data, rapport) när de finns; eventuellt WebAssembly om 15M känns seg på mobil.

## 21. Fas 5a: repo, GitHub och Hugging Face, 2026-11-24

**Git och GitHub.** Projektmappen blev ett git-arkiv (den hade inte varit det) med första commit "Pytte: tiny Swedish language models trained from scratch on a MacBook" och ligger som **privat** repo på `https://github.com/dragon6sic6/pytte` tills vidare. Med i repot: koden, skripten, mätningarna (`eval/`), varumärket, webbdemon med de tolv modellfilerna (59 MB i vanlig git, inte LFS, eftersom GitHub Pages inte serverar LFS-filer och LFS-bandbredden är begränsad), README och den här loggen. Utanför (`.gitignore`): `checkpoints/`, loggar, det kompilerade `run`, `.claude/` och de tre research-rapporterna som var underlag för planen. Inga hemligheter i historiken (kontrollerat med grep innan första commit).

**Licenser (beslut).** Kod och modeller **Apache-2.0** (samma som Gemma 4, vars villkor inte begränsar utdata, se avsnitt 10), SagoSvenska **CC BY 4.0**. `LICENSE` är Apache-texten.

**Hugging Face (förberett, väntar på inloggning).** `scripts/make_model_cards.py` bygger tolv uppladdningsmappar i `/Volumes/Modelltest/pytte/hf/` (safetensors, config, tokenizer.json, int8-.bin och ett modellkort genererat ur summary/eval/judge/eval_pairs: arkitektur, träning, resultattabell, filer, användning i C och Python, begränsningar, licens). `scripts/hf_upload.py --models --dataset` skapar privata repon `<användare>/pytte-<storlek>[-chat]` och datasetrepot `<användare>/sagosvenska` (mapparna `v1.1/`, `v2/`, `mix1/` med jsonl och stats, datasetkort med tabell över de tre seten, varför de finns, rensningen och licensen); `--public` vid lansering. Kräver `hf auth login` med en skrivtoken, vilket bara Mathias kan göra; sedan är uppladdningen ett kommando (modellerna 310 MB, datasetet 211 MB).

**Kvar i fas 5.** Rapporten (avsnitt 6, fas 5), Zenodo-DOI, länkarna i demons sidfot, modellkortens länk till rapporten.

## 22. Fas 5b: rapporten, 2026-11-25

**Beslut (Mathias, 2026-11-25):** rapporten skrivs före Hugging Face-uppladdningen, så att modell- och datasetkorten kan länka till den.

**Resultat:** `paper/pytte.pdf`, 19 sidor A4, engelska med svensk sammanfattning. Titel: *Pytte: Tiny Language Models That Write Coherent Swedish, Trained from Scratch on a Laptop*. Mathias Karlsson ensam författare, med ett eget avsnitt (9) om vad AI-assistenten gjorde, vad författaren beslutade och vilka misstag som begicks. Struktur: sammanfattning, inledning med bidrag, tidigare arbete, data (generatorval, recept v1, mallfyndet, recept v2, de tre seten, `<image|>`-buggen), modell och träning, mätmetod med kvalitetsgolv och brusanalys, resultat (v1.1-familjen, tvillingarna, de tre datablandningarna, chattfamiljen med överanpassning och x3-testet, export/run.c/webbläsare), diskussion, begränsningar, reproducerbarhet och kostnad, slutsats, referenser, bilagor (de 20 fynden, oselekterade exempel, grammatik per kategori).

**Hur den byggs (allt i `paper/`, `make paper`).** `collect_results.py` läser alla 36 modellers summary/eval/judge/eval_pairs och chattmodellernas valideringskurvor till `results.json` (kontrollerad mot loggens tabeller). `make_figures.py` ritar sex figurer som SVG utan beroenden (parameterbudget, skalning sv/en/en8m, tre datablandningar, grammatik per kategori, instruktionsföljning, överanpassningen i 15M-chat) med en färgblindhetskontrollerad palett, 2 px linjer, ändetiketter och förklaring. `build.py` fyller tabellerna i `pytte.src.html` ur `results.json` (familjetabell, tvillingar, blandningar, släppfamilj, chattfamilj, x1 mot x3, kategorier, exempelsagor) och skriver PDF med Chrome i headless-läge. Ingen LaTeX finns på maskinen, därav HTML-vägen; typsnitt STIX Two Text. Siffror i löptexten är hämtade ur den här loggen.

**Ärlighetspunkter som rapporten själv tar upp.** Kvalitetsgolvet anges som det formulerades i avsnitt 4 (domare ≥ 8 av 10 på grammatik och sammanhang, minimala par ≥ 70 %), med noten att 50–100 exempel bedömdes i stället för 200 och att inget blindtest med människor gjordes. Med det golvet är 1M den minsta modellen som klarar det; 0,3M klarar grammatik och minimala par men inte sammanhang. Exemplen i bilaga B är det första av varje modells utvärderingsexempel, och de två första småpratsexemplen i grupperna "vem" och "utanför" råkar vara de två av 24 som modellen svarade fel på; de står kvar.

**Följdändringar.** README:s statusruta (som fortfarande sa 9 november och "0,3M skriver sammanhängande svenska") är uppdaterad och pekar på rapporten; modellkorten länkar till den (`scripts/make_model_cards.py`, korten omgenererade).

**Kvar.** Mathias läser igenom; Zenodo-DOI vid lansering (läggs in på förstasidan och i korten); sedan Hugging Face-uppladdningen (avsnitt 21).

## 23. Ändrad publiceringsväg: hobbyprojekt på mathiaskarlsson.se/pytte, 2026-11-25

**Beslut (Mathias, 2026-11-25).** Modellerna har varit ett test av om det här går. I stället för en modellsläpp på Hugging Face publiceras Pytte som ett **hobbyprojekt på hans egen sajt**, med en egen sida `mathiaskarlsson.se/pytte` och ett kort under `/sidoprojekt`, och med öppenhet om att han inte vet hur bra det är. Sidan ska rymma allt: demon, berättelsen, rapporten och modellfilerna för nedladdning. Det ersätter planens lansering tills vidare; Hugging Face och Zenodo är inte aktuella nu men blockeras inte av något.

**Vad som följer av det.** (1) Sajten är ett Next.js-projekt (`mathiaskarlsson2026final`, svenska och engelska, innehåll i `content/`, Vercel), som ägs av en annan session. Sidan byggs därför där, i sajtens egen design och struktur, från ett överlämningspaket härifrån: `site/BRIEF.md` (beställning: sidans delar i ordning, filer med absoluta sökvägar, hur demon fungerar för en React-portning, vad som ska avgöras på sajtsidan, verifieringspunkter, vad som inte ingår) och `site/content.pytte.ts` (all text på svenska och engelska: kort, hero, demons strängar, varför, siffror, fem lärdomar, sex punkter under "Vad jag inte vet", nedladdningar). Körkoden (`web/pytte.js`, `web/worker.js`) och modellfilerna (`web/models/`, 59 MB) återanvänds som de är. (2) Rapporten fick hobbyprojekt-inramning: en rad på förstasidan (oberoende hobbyprojekt, inte granskad), en mening i sammanfattningen och inledningen, en ny första punkt under begränsningar (inte granskad, inte jämförd med andras modeller) och motsvarande i den svenska sammanfattningen; förstasidan pekar nu på `mathiaskarlsson.se/pytte`. (3) Stora nedladdningar (fulla vikter 310 MB, datasetet 211 MB) läggs som GitHub-releaser när repot blir publikt; modellkorten för Hugging Face ligger kvar på disken men används inte. (4) Inget publiceras förrän Mathias sett sidan.

**Öppna frågor för sajtsidan.** Var .bin-filerna serveras ifrån (59 MB statiskt, 31 MB om en besökare drar reglaget genom alla storlekar; bandbredden på hostingplanen behöver kontrolleras före lansering), när GitHub-repot görs publikt, och OG-bild/kort i sajtens stil.

**Sidan byggd (2026-11-25, rapport från sajtsessionen, kontrollerad härifrån).** Grenen `pytte` i `mathiaskarlsson2026final` har två commits som inte finns på `origin/master`: `/pytte` och `/en/pytte` i sajtens design med briefens sektionsordning, kort på `/sidoprojekt` och på startsidan, delningsbild, sitemap. `pytte.js` och `worker.js` ligger oförändrade i `public/pytte/` (identiska filer), de tolv modellerna i `public/pytte/models/` och rapporten som `public/pytte/pytte.pdf` (samma fil som `paper/pytte.pdf`). Lokal förhandsvisning på `localhost:3001/pytte` svarar på båda språken och serverar modellfilerna. Briefens tre kontroller går igenom enligt sajtsessionen (frö 624419 på 2M ger den kända sagan). GitHub-länk, run.c och data visar "Kommer snart" bakom en flagga tills repot är publikt. Inget är pushat eller publicerat; hosting och repots synlighet ligger hos Mathias.

**Repot publikt (2026-11-25).** `github.com/dragon6sic6/pytte` är publikt sedan i dag, med ärenden avstängda; avsnitt 21 skrevs medan det var privat. På sajtens `pytte`-gren pekar kod- och run.c-länkarna nu på GitHub, medan träningsdatan fortfarande står som kommande. Sajten är inte publicerad ännu.

**Träningsdatan släppt (2026-11-25, rapport från sajtsessionen; beslutet är Mathias).** SagoSvenska ligger som GitHub-släpp `sagosvenska-1.1` (https://github.com/dragon6sic6/pytte/releases/tag/sagosvenska-1.1): `sagosvenska.zip`, 49 MB, med mapparna `v1.1/`, `v2/` och `mix1/` (train/val/test.jsonl och stats.json i varje) och en README byggd på datasetkortet i `scripts/hf_upload.py`, CC BY 4.0. Sajtsessionen uppger att delningarna stämmer mot rapportens tabell 2 och att en ordsökning efter våld och skräp inte gav något. Härifrån är kontrollerat att släppet finns, är publicerat (inte utkast), att filen är 49 301 322 byte och att taggen pekar på `main`; zip-filens innehåll är inte öppnat härifrån. README pekar nu på släppet, och licensraden där är rättad: den sa "Code: MIT" och "to be confirmed", men filen `LICENSE` i repot är Apache-2.0, rapporten och sidan säger Apache-2.0 för kod och modeller, och datan är CC BY 4.0.

**Ton i README, rapport och skript (2026-11-25, på Mathias uppdrag).** Meningen om att Pytte "så vitt vi vet är det första öppna, dokumenterade försöket" är struken ur README och ur rapportens avsnitt 1, och omdömet om Linköpings-kursens publicering ("utan licens, modellkort eller rapport") är ersatt med fakta: vad kursen publicerade, och att datan saknar licens och därför inte kunde återanvändas. De två oanvända Hugging Face-skripten säger inte längre "the smallest open, documented models" och har fått en rad om att de inte används. Rapporten är ombyggd: 19 sidor som förut, bara sida 2 skiljer sig. Sajtens kopia av PDF:en behöver bytas av sajtsessionen.

**Sidan är ute (kontrollerat 2026-11-25).** `mathiaskarlsson.se/pytte` och `/en/pytte` svarar, kortet finns på `/sidoprojekt`, och sajtens `master` ligger på samma commit som grenen `pytte`. Publiceringen gjordes av sajtsessionen; tidigare rader i det här avsnittet som säger att sajten inte är publicerad gällde när de skrevs. Kontrollerat härifrån mot den publika sajten: rapporten där är samma fil som `paper/pytte.pdf` i repot, `pytte.js` är identisk med `web/pytte.js`, tre stickprov av modellfilerna har rätt storlek (0,3M, 7M-chat, 15M), och sidan länkar till GitHub-repot och till datasläppet `sagosvenska-1.1`. Demon är inte provkörd i en webbläsare mot den publika sajten härifrån.

**Demons exempel bytta (2026-11-25).** Mathias provkörde den publika sidan med "Det var en gång en räv som vill flyga" och fick en saga där räven försvann efter två meningar. Orsaken är datan: alla träningssagor handlar om ett namngivet barn i vardagen, och sidans egna exempel (en älg, en räv som vill flyga) ledde besökaren utanför den. Prov med demons inställningar (temp 0,8, top-p 0,95, 20 frön per modell, `run.c` på int8-filerna): med en djurbörjan dyker ett barnnamn upp i 40–80 % av sagorna och djuret nämns sällan mer än en gång ("en liten älg": 0–15 % över storlekarna); med "Alma bor i en liten stuga i skogen. En dag" finns Alma kvar i 100 % från 1M (65 % vid 0,3M). Chattmodellerna (3M, 7M, 15M) fick med både räv och flygande i 0–10 % av svaren på "Berätta en saga om en räv som vill flyga.", men namn, katt och gård i 80–95 % på "Berätta en saga om Freja och en katt som utspelar sig på en bondgård." Måtten är ordträffar, inte en bedömning av sagorna. Bytt i `web/index.html` och `site/content.pytte.ts`: startmeningen, chattexemplet, och en ny rad som säger att sagorna ofta blir konstiga och varför, och som uppmanar besökaren att prova räven. Samtidigt är "från noll" bytt mot "från grunden" i demon och sidtexten, som på sajten. Sajtsessionen lägger in texterna på sidan.

## 24. Inuti Pytte: en sida som visar vad modellen räknar, 2026-11-25

**Uppdrag (Mathias).** En sida som verkligen lär ut hur modellen fungerar, "avancerad" och med wow-upplevelse, där allt ska stämma. Byggd som `web/inuti.html`: man skriver en mening, modellen körs i webbläsaren, och sidan visar modellens egna mellanresultat i sex steg (token, inbäddning, attention, lager för lager, slumpen, träningen). `web/arkitektur.html` är ritningen över delarna, med parametrar per del för alla sex storlekar (uträknade ur konfigurationen; summan stämmer exakt med modellerna).

**Motorn.** `pytte.js` har fått en "tap": sätts `model.tap = {}` före `forward()` kopieras inbäddningen, attention-vikterna per lager och huvud, storleken på varje lagers två tillägg och tillståndet efter varje lager. Aritmetiken är orörd. `worker.js` har två nya meddelanden, `trace` och `neighbors`. Gissningen "efter lager k" fås genom att skicka mellantillståndet genom sista normaliseringen och utmatningstabellen; modellen är inte tränad att svara så, och sidan säger det. Sidan påstår ingenting om vad enskilda huvuden eller lager "gör", bara vad som syns.

**Kontroll.** Motorn före och efter ändringen: 70 av 70 tokenföljder identiska (7 modeller, 2 början, 5 frön, 250 token). Attention-raderna summerar till 1, och gissningen efter sista lagret är exakt modellens utdata (största skillnad 0). Sidan är provkörd i webbläsaren med 0,3M, 1M, 7M och 15M, i mobilbredd och skrivbordsbredd, och den vanliga demon fungerar som förut.

**Fynd 21: webbläsaren och run.c ger inte samma saga, annat än i början.** Samma kontroll jämförde också JavaScript mot `run.c`: 20 av 70 sagor var identiska hela vägen. Alla 70 börjar lika, och den första skillnaden kommer efter 32 till 573 byte. Påståendet i avsnitt 20, README och rapporten ("byte för byte samma saga") byggde på en enda saga på 81 token och var fel. Det är rättat på alla tre ställen. Trolig orsak, inte bekräftad: JavaScript räknar i 64-bitars flyttal och C i 32-bitars, så att en avrundad aktivering då och då hamnar på andra sidan en gräns. C mot Python (avsnitt 19) berörs inte.

**Träningen på sidan är en riktig körning.** `pytte/train.py` har fått `--snapshots` och `--snap-prompt`, som skriver ett textprov vid valda steg. pytte-1m tränades om på mix1 med samma inställningar som släppmodellen (8 minuter, lr 0,002): 10 147 steg, 83,1M token, bästa valideringsförlust 1,509 mot släppmodellens 1,508. Proverna (`web/inuti-training.json`): vid 0 s slumpade token, vid 5 s (förlust 5,5) de första punkterna, vid 10 s (3,9) korta meningar, vid 77 s (1,9) första provet som nämner Alma. Ett prov per tidpunkt, draget med slump, så det är en illustration och inte en mätning. Körningen ligger i `checkpoints/pytte-1m-sv3-snap`.

**Ute på sajten samma dag.** Sajtsessionen lade ut `inuti.html`, en engelsk version (`inuti.en.html`, skapad ur den svenska av `scripts/make_inuti_en.py`), arkitekturbilden som helsida och som inbäddad bild på Pytte-sidan (`arkitektur-embed*.html`, skapade av `scripts/make_arkitektur.py`), och den rättade rapporten. Kontrollerat härifrån: filerna på sajten är identiska med repots.

**Sammanslagen och omklädd (på Mathias uppdrag).** `inuti.html` har nu arkitekturbilden inbyggd som en levande karta: en fast kolumn på breda skärmar, ett kort sist på smala. Kartan visar besökarens egen mening, token och sannolikheter, markerar den del som hör till steget man läser, och ett klick på en del leder till steget. Varje steg avslutas med delens storlek i den valda modellen (parametrarna räknas ur konfigurationen, som i arkitekturbilden). Sidan har sajtens färger och typsnitt och följer sajtens temaval.

**Fynd 22: en kapplöpning i `worker.js` kunde ge en rörig saga.** Startades en saga om medan en annan skrevs på samma modell (knappen "Skriv sagan" eller "Nytt frö" mitt i en saga, eller ett nytt chattmeddelande), nollställde den nya körningen flaggan `cancelled` innan den gamla slingan hunnit se den. Två slingor skrev då i samma modells cache, och den nya sagan blev osammanhängande. Det syntes som "En dag går hon ut. dag går hon ut ser en stor, är himl" när jag klickade under en pågående saga. Byte av modellstorlek berördes inte, eftersom varje storlek har sin egen modell. Rättat med ett löpnummer per körning: en ny körning, ett avbrott eller en `trace`/`probs` stoppar den äldre vid nästa paus. Prov: tre omstarter i snabb följd ger nu samma saga som en ostörd körning med samma frö, och den börjar som `run.c`. Buggen har funnits sedan demon byggdes (avsnitt 20) och fanns därmed också på sajten.

**Kvar.** Den sammanslagna sidan och den rättade `worker.js` ligger i repot men inte på sajten.

## Källor som verifierades i den här genomgången

- Hugging Face: `tdde19-llm-from-scratch/TinyStoriesSwe` och tillhörande modeller, `qwrt/Swedish0.1M`, `Dxniz/TinyStories-Multilingual` (API-anrop 2026-11-06)
- PyPI: mlx 0.32.4, torch 2.15.0, tokenizers 0.23.3 (wheels för cp314 macOS arm64)
- [Language Model Builder (NYU Shanghai RITS)](https://rits.shanghai.nyu.edu/ai/language-model-builder-train-a-small-llm-from-scratch-on-your-mac/), [Flavio Copes](https://flaviocopes.com/language-model-builder/)
- [Karpathy microgpt](https://karpathy.github.io/2026/02/12/microgpt/) via [beri.net](https://www.beri.net/learning/karpathy-microgpt) och [aktagon](https://signals.aktagon.com/articles/2026/03/microgpt-a-200-line-implementation-of-gpt-from-scratch/)
- [Sean Goedecke: What's the strongest AI model you can train on a laptop in five minutes?](https://www.seangoedecke.com/model-on-a-mbp/)
- [TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759)
- [SweCTRL-Mini](https://arxiv.org/pdf/2304.13994), [GPT-SW3](https://arxiv.org/html/2305.12987v3)
- Ej verifierade påståenden från deep-research-rapporten: StateHead (BabyLM 2026), "TinyLM 78K". Behandla som okända.

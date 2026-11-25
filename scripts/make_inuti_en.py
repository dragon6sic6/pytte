"""Write web/inuti.en.html from web/inuti.html by replacing every Swedish string with its English one.

    python3 scripts/make_inuti_en.py

The Swedish page is the source. If a string below is no longer found there, the script stops and says which,
so the two pages cannot drift apart silently.
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
ALL = [  # replaced wherever they occur
    ('.replace(".", ",")', ""), ('"sv-SE"', '"en-US"'), ("Inuti Pytte", "Inside Pytte"),
]
ONCE = [
    ('<html lang="sv">', '<html lang="en">'),
    ("Skriv en mening och se vad som händer inuti en liten språkmodell: token, inbäddning, attention, lager för lager, och slumpen som väljer nästa ord. Allt räknas i din webbläsare.",
     "Type a sentence and see what happens inside a tiny language model: tokens, embedding, attention, layer by layer, and the chance that picks the next word. Everything is computed in your browser."),
    ("Skriv en mening. Allt du ser längre ner är vad modellen faktiskt räknar ut för just den meningen, i din webbläsare, medan du tittar.",
     "Type a sentence. Everything further down is what the model actually computes for that sentence, in your browser, while you watch. Pytte only knows Swedish: the sentence below means “Alma lives in a small”."),
    ('aria-label="Din mening"', 'aria-label="Your sentence"'), ('aria-label="Modellstorlek"', 'aria-label="Model size"'), ('aria-label="Stegen"', 'aria-label="The steps"'),
    ('data-name="Token"', 'data-name="Tokens"'), ('data-name="Tal"', 'data-name="Numbers"'), ('data-name="Lager"', 'data-name="Layers"'),
    ('data-name="Slumpen"', 'data-name="Chance"'), ('data-name="Träningen"', 'data-name="Training"'),
    # 1
    ("<h2>Texten blir bitar</h2>", "<h2>The text becomes pieces</h2>"),
    ("Modellen ser inga bokstäver. Den ser <b>token</b>: bitar av ord som den har en nummerlapp för. Vanliga ord är en bit, ovanliga delas i flera. Mellanslaget hör till ordet efter.",
     "The model does not see letters. It sees <b>tokens</b>: pieces of words that each have a number. Common words are one piece, rare ones are split into several. The space belongs to the word after it."),
    ("Först i raden står ett starttoken. Det sätts in före varje saga under träningen och betyder ”här börjar en ny text”.",
     "First in the row is a start token. It is put before every story during training and means “a new text begins here”."),
    # 2
    ("<h2>Varje bit blir en rad med tal</h2>", "<h2>Each piece becomes a row of numbers</h2>"),
    ("Varje token byts mot sin rad i en tabell. Raden är modellens beskrivning av vad token betyder. Här är raderna för din mening, en ruta per tal: blått är negativt, orange positivt.",
     "Each token is swapped for its row in a table. The row is the model's description of what the token means. Here are the rows for your sentence, one cell per number: blue is negative, orange positive."),
    ("<span>negativt</span><i></i><span>positivt</span>", "<span>negative</span><i></i><span>positive</span>"),
    ("Talen säger ingenting var för sig. Men token som används på liknande sätt får rader som pekar åt liknande håll. Klicka på ett token ovan så letar jag upp dem som ligger närmast i tabellen.",
     "The numbers say nothing one by one. But tokens that are used in similar ways get rows that point in similar directions. Click a token above and the page finds the ones closest to it in the table."),
    ('id="nb-title">Närmast i tabellen</h3>', 'id="nb-title">Closest in the table</h3>'),
    ("Siffran är likheten mellan raderna, där 1 är samma riktning och 0 är ingen likhet alls. Ingen har talat om för modellen vilka ord som hör ihop. Det har den fått fram ur sagorna.",
     "The number is the similarity between the rows, where 1 is the same direction and 0 is no similarity at all. Nobody told the model which words belong together. It worked that out from the stories."),
    # 3
    ("<h2>Bitarna tittar på varandra</h2>", "<h2>The pieces look at each other</h2>"),
    ("Det här är <b>attention</b>, det enda stället där token utbyter information. Välj ett token. Färgen på de andra visar hur mycket det valda tittar på dem. Det kan bara titta bakåt, aldrig framåt.",
     "This is <b>attention</b>, the only place where tokens exchange information. Pick a token. The colour of the others shows how much the chosen one looks at them. It can only look backwards, never ahead."),
    ('id="pick-layer"><b>Lager</b>', 'id="pick-layer"><b>Layer</b>'), ('id="pick-head"><b>Huvud</b>', 'id="pick-head"><b>Head</b>'),
    ("Varje lager har flera <b>huvuden</b> som tittar efter olika saker samtidigt. Byt lager och huvud och se hur mönstret ändras. Några tittar mest på ordet precis före, några på starttoken, några långt bak i meningen.",
     "Every layer has several <b>heads</b> that look for different things at the same time. Change layer and head and watch the pattern change. Some look mostly at the word just before, some at the start token, some far back in the sentence."),
    ("Alla huvuden på en gång, för det valda token</h3>", "All heads at once, for the chosen token</h3>"),
    ("En rad per lager, en grupp per huvud, en ruta per token i meningen. Mörkare ruta betyder mer uppmärksamhet. Klicka på en grupp för att välja den.",
     "One row per layer, one group per head, one cell per token in the sentence. A darker cell means more attention. Click a group to select it."),
    ("Vad ett visst huvud ”letar efter” har jag inte undersökt. Bilden visar vart det tittar, inte varför.",
     "I have not investigated what a particular head is “looking for”. The picture shows where it looks, not why."),
    # 4
    ("<h2>Svaret växer fram, lager för lager</h2>", "<h2>The answer takes shape, layer by layer</h2>"),
    ("Efter attention bearbetas varje token för sig i ett litet nätverk, <b>feed-forward</b>. Sedan kommer nästa lager och gör samma två saker igen. Tabellen visar två saker för meningens sista token: hur mycket varje lager ändrade, och vad modellen skulle ha gissat om den hade fått sluta där.",
     "After attention each token is processed on its own in a small network, the <b>feed-forward</b> part. Then the next layer does the same two things again. The table shows two things for the last token of the sentence: how much each layer changed, and what the model would have guessed had it stopped there."),
    ("</i>ändring från attention</span>", "</i>change from attention</span>"), ("</i>ändring från feed-forward</span>", "</i>change from feed-forward</span>"),
    ("</i>grön gissning = den som till sist vinner</span>", "</i>green guess = the one that wins in the end</span>"),
    ("Gissningen mitt i modellen får man genom att skicka mellanresultatet direkt till sista steget. Modellen är inte tränad att svara så, så de tidiga raderna ska läsas som en fingervisning. Sista raden är modellens riktiga svar. Första raden är nästan alltid samma token en gång till: utan några lager finns det inget annat att gå på än tokenet självt.",
     "The guess in the middle of the model comes from sending the intermediate result straight to the last step. The model is not trained to answer that way, so read the early rows as a hint. The last row is the model's real answer. The first row is almost always the same token once more: without any layers there is nothing to go on but the token itself."),
    # 5
    ("<h2>Slumpen väljer nästa bit</h2>", "<h2>Chance picks the next piece</h2>"),
    ("Ut kommer inte ett ord utan en sannolikhet för vart och ett av modellens token. Sedan dras ett av dem. Två reglage styr dragningen. Dra i dem och se vad som händer med chanserna.",
     "What comes out is not a word but a probability for every one of the model's tokens. Then one of them is drawn. Two sliders control the draw. Move them and see what happens to the odds."),
    ('<label for="temp">Temperatur</label>', '<label for="temp">Temperature</label>'),
    ('id="draw">Dra ett token</button>', 'id="draw">Draw a token</button>'), ('id="auto">Skriv 15 till</button>', 'id="auto">Write 15 more</button>'), ('id="reset">Börja om</button>', 'id="reset">Start over</button>'),
    ("<b>Låg temperatur</b> gör de troliga ännu troligare, och sagan blir förutsägbar. <b>Hög temperatur</b> jämnar ut chanserna, och sagan blir vildare. <b>Top-p</b> stryker de minst troliga helt, så att ett riktigt olyckligt drag inte förstör meningen. Varje gång du drar ett token läggs det till meningen, och allt på den här sidan räknas om.",
     "<b>Low temperature</b> makes the likely ones even more likely, and the story becomes predictable. <b>High temperature</b> evens out the odds, and the story gets wilder. <b>Top-p</b> removes the least likely ones altogether, so that one really unlucky draw does not ruin the sentence. Every time you draw a token it is added to the sentence, and everything on this page is computed again."),
    # 6
    ("<h2>Så lärde den sig</h2>", "<h2>How it learned</h2>"),
    ("Från början är alla tal i modellen slump. Träningen är samma väg som ovan, miljontals gånger: modellen gissar nästa token i en riktig saga, får se rätt svar, och alla tal knuffas en liten bit åt det håll som hade gett en bättre gissning.",
     "At the start every number in the model is random. Training is the same path as above, millions of times: the model guesses the next token in a real story, is shown the right answer, and every number is nudged a little in the direction that would have given a better guess."),
    ("Jag tränade om den minsta modellen som skriver sammanhängande, med en miljon tal, och sparade vad den skrev under tiden. Dra i reglaget.",
     "I retrained the smallest model that writes coherently, with one million numbers, and saved what it wrote along the way. Move the slider. The texts are in Swedish; watch them go from noise to sentences."),
    ('aria-label="Tid in i träningen"', 'aria-label="Time into training"'),
    ('aria-label="Förlusten sjunker snabbt de första sekunderna och sedan långsamt"', 'aria-label="The loss falls quickly in the first seconds and then slowly"'),
    ("<b>Förlusten</b> mäter hur förvånad modellen blir av rätt svar. Vid ren gissning bland 4&nbsp;096 token är den 8,3. I proverna här dyker punkterna upp efter fem sekunder och hela meningar efter tio. Efter drygt en minut handlar texten om Alma. Resten av tiden går åt till det svåra: att hålla ihop en hel saga.",
     "<b>The loss</b> measures how surprised the model is by the right answer. For a pure guess among 4,096 tokens it is 8.3. In the samples here full stops appear after five seconds and whole sentences after ten. After a little over a minute the text is about Alma. The rest of the time goes to the hard part: holding a whole story together."),
    ("Allt på den här sidan räknas av samma program som i demon, med samma modellfiler. Inget skickas någonstans. Vill du se delarna som en ritning finns <a href=\"arkitektur.html\">arkitekturbilden</a>. <a href=\"inuti.en.html\">This page in English</a>.",
     "Everything on this page is computed by the same program as the demo, with the same model files. Nothing is sent anywhere. <a href=\"inuti.html\">Den här sidan på svenska</a>."),
    # script
    ("`${label} parametrar`", "`${label} parameters`"), ('"räknar…"', '"computing…"'),
    ("· ${m.T - 1} token · ${m.L} lager · ${m.H} huvuden per lager · ${m.D} tal per token · räknat på ${Math.round(performance.now() - t0)} ms`",
     "· ${m.T - 1} tokens · ${m.L} layers · ${m.H} heads per layer · ${m.D} numbers per token · computed in ${Math.round(performance.now() - t0)} ms`"),
    ('" · sidan visar högst 39 token"', '" · the page shows at most 39 tokens"'),
    ('`token nummer ${tk.id}` : "starttoken"', '`token number ${tk.id}` : "start token"'),
    ('`${words} ord blev ${t.T - 1} token. Modellen har ${t.vocabSize.toLocaleString("en-US")} olika token att välja bland.`',
     '`${words} words became ${t.T - 1} tokens. The model has ${t.vocabSize.toLocaleString("en-US")} different tokens to choose from.`'),
    ("`· ${t.D} tal per token`", "`· ${t.D} numbers per token`"),
    ("`Närmast ”${show(tk.text)}” i tabellen`", "`Closest to “${show(tk.text)}” in the table`"),
    ('`${pct(w)} av uppmärksamheten` : "kommer senare, syns inte"', '`${pct(w)} of the attention` : "comes later, cannot be seen"'),
    ('" tittar mest på "', '" looks mostly at "'),
    ('s ? `Efter lager ${s}` : "Bara inbäddningen"', 's ? `After layer ${s}` : "Embedding only"'),
    ('"hur mycket lagret ändrade tokenets tal"', '"how much the layer changed the token\'s numbers"'),
    ('g.id === t.eot ? "slut" :', 'g.id === t.eot ? "end" :'),
    ('id === t.eot ? "slut på sagan" :', 'id === t.eot ? "end of story" :'), ('kept ? pct(prob) : "struken"', 'kept ? pct(prob) : "removed"'),
    ('`${d.keep.toLocaleString("en-US")} av ${t.logits.length.toLocaleString("en-US")} token är kvar i dragningen. De tolv troligaste visas.`',
     '`${d.keep.toLocaleString("en-US")} of ${t.logits.length.toLocaleString("en-US")} tokens are left in the draw. The twelve most likely are shown.`'),
    ('"Modellen drog ”slut på sagan”. Den tycker att sagan är klar."', '"The model drew “end of story”. It thinks the story is finished."'),
    ('"Sidan visar högst 39 token. Tryck på Börja om för en ny mening."', '"The page shows at most 39 tokens. Press Start over for a new sentence."'),
    ('`ren gissning: ${guess.toFixed(1)}`', '`pure guess: ${guess.toFixed(1)}`'), ('}, "förlust"));', '}, "loss"));'),
    ('s < 1 ? "0 sekunder" : s < 90 ? `${Math.round(s)} sekunder` : `${(s / 60).toFixed(1)} minuter`', 's < 1 ? "0 seconds" : s < 90 ? `${Math.round(s)} seconds` : `${(s / 60).toFixed(1)} minutes`'),
    ('token lästa · förlust ${loss.toFixed(2)}` : "inget läst än, alla tal är slump"', 'tokens read · loss ${loss.toFixed(2)}` : "nothing read yet, every number is random"'),
    ('`En riktig körning på en MacBook: pytte-1m, ${TR.params.toLocaleString("en-US")} parametrar, ${TR.steps.toLocaleString("en-US")} steg på ${String(TR.minutes)} minuter, ${Math.round(TR.tokens / 1e6)} miljoner token lästa. Tidsaxeln är hoptryckt så att de första sekunderna syns. Texterna är dragna med slump och hade blivit annorlunda med ett annat frö.`',
     '`A real run on a MacBook: pytte-1m, ${TR.params.toLocaleString("en-US")} parameters, ${TR.steps.toLocaleString("en-US")} steps in ${String(TR.minutes)} minutes, ${Math.round(TR.tokens / 1e6)} million tokens read. The time axis is compressed so that the first seconds show. The texts are drawn at random and would have come out differently with another seed.`'),
]


def main():
    s = (WEB / "inuti.html").read_text(encoding="utf-8")
    for a, b in ALL:
        assert a in s, f"not found: {a}"
        s = s.replace(a, b)
    missing = [a for a, _ in ONCE if s.count(a) != 1]
    if missing:
        raise SystemExit("these Swedish strings are not found exactly once in web/inuti.html:\n  " + "\n  ".join(m[:110] for m in missing))
    for a, b in ONCE:
        s = s.replace(a, b)
    (WEB / "inuti.en.html").write_text(s, encoding="utf-8")
    print("web/inuti.en.html")


if __name__ == "__main__":
    main()

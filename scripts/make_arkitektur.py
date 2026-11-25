"""Write the other three versions of the architecture diagram from web/arkitektur.html (Swedish, full page):

    web/arkitektur.en.html          English, full page
    web/arkitektur-embed.html       Swedish, the big diagram inline (for an iframe)
    web/arkitektur-embed.en.html    English, inline

    python3 scripts/make_arkitektur.py

The Swedish page is the source. If a string below is no longer found there, the script stops and says which.
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
ALL = [
    ('.replace(".", ",")', ""), ('"sv-SE"', '"en-US"'), ("Så fungerar Pytte", "How Pytte works"), ('t === "Lager"', 't === "Layers"'),
    (">Normalisering</text>", ">Normalisation</text>"), (">hopplänk</text>", ">skip connection</text>"), ('aria-label="Normalisering"', 'aria-label="Normalisation"'),
]
ONCE = [
    ('<html lang="sv">', '<html lang="en">'),
    ('id="back" href="../pytte"', 'id="back" href="../en/pytte"'),
    ('href="inuti.html" style="font-size:15px;text-decoration:none">Inuti Pytte</a>', 'href="inuti.en.html" style="font-size:15px;text-decoration:none">Inside Pytte</a>'),
    ("Pytte gör en enda sak: gissar nästa bit text. Bilden visar vägen från det du skriver till den gissningen. Klicka på den för att se varje del, vad den gör och hur stor den är.",
     "Pytte does one thing: it guesses the next piece of text. The picture shows the path from what you write to that guess. Click it to see every part, what it does and how big it is."),
    ("Text blir token, token blir tal, talen går genom lagren och ut kommer sannolikheter för nästa token. Det valda tokenet läggs till texten och allt körs igen.",
     "Text becomes tokens, tokens become numbers, the numbers pass through the layers and out come probabilities for the next token. The chosen token is added to the text and everything runs again."),
    ("det valda tokenet läggs till texten, och så ett varv till", "the chosen token is added to the text, and round it goes again"),
    ('<span class="open">Öppna hela bilden</span>', '<span class="open">Open the full picture</span>'),
    ("Samma väg i alla sex modeller. Bara storleken på delarna skiljer.", "The same path in all six models. Only the size of the parts differs."),
    ('aria-label="Modellstorlek"', 'aria-label="Model size"'), ('id="close">Stäng</button>', 'id="close">Close</button>'),
    ("Arkitekturen uppifrån och ner: text, tokeniserare, inbäddning, lagren med normalisering, attention och feed-forward, sista normalisering, utmatning, sannolikheter och val av nästa token.",
     "The architecture from top to bottom: text, tokenizer, embedding, the layers with normalisation, attention and feed-forward, the final normalisation, output, probabilities and the choice of the next token."),
    ("läggs till texten, sedan ett varv till</text>", "added to the text, then one more round</text>"),
    ("samma tabell används igen</text>", "the same table is used again</text>"),
    ('aria-label="Texten"', 'aria-label="The text"'), ('aria-label="Tokeniserare"', 'aria-label="Tokenizer"'), ('aria-label="Inbäddning"', 'aria-label="Embedding"'),
    ('aria-label="Lagren"', 'aria-label="The layers"'), ('aria-label="Hopplänk"', 'aria-label="Skip connection"'), ('aria-label="Sista normaliseringen"', 'aria-label="The final normalisation"'),
    ('aria-label="Utmatning"', 'aria-label="Output"'), ('aria-label="Sannolikheter"', 'aria-label="Probabilities"'), ('aria-label="Slumpen väljer"', 'aria-label="Chance picks"'),
    (">Tokeniserare</text>", ">Tokenizer</text>"), (">Inbäddning</text>", ">Embedding</text>"),
    (">varje token tittar bakåt i texten</text>", ">each token looks back in the text</text>"), (">och hämtar det den behöver</text>", ">and fetches what it needs</text>"),
    (">varje token bearbetas för sig</text>", ">each token on its own</text>"), (">genom ett litet nätverk</text>", ">through a small network</text>"),
    (">Utmatning</text>", ">Output</text>"), (">Sannolikheter</text>", ">Probabilities</text>"), (">Slumpen väljer</text>", ">Chance picks</text>"),
    # numbers written with a decimal comma
    ('label: "0,3 M"', 'label: "0.3 M"'), ('mb: "0,38"', 'mb: "0.38"'), ('mb: "1,4"', 'mb: "1.4"'), ('mb: "2,4"', 'mb: "2.4"'), ('mb: "3,6"', 'mb: "3.6"'), ('mb: "7,3"', 'mb: "7.3"'),
    ('items = ["Text", "Token", "Tal", "Lager", "Sannolikheter", "Nästa token"]', 'items = ["Text", "Tokens", "Numbers", "Layers", "Probabilities", "Next token"]'),
    ("`varje token blir ${m.dim} tal`", "`every token becomes ${m.dim} numbers`"), ("`Lager 1 av ${m.layers}`", "`Layer 1 of ${m.layers}`"),
    ("`${m.layers} likadana efter varandra`", "`${m.layers} alike, one after another`"), ("`${m.heads} huvuden med ${m.hd} tal var`", "`${m.heads} heads with ${m.hd} numbers each`"),
    ("`${m.dim} tal → ${n(m.ff)} → ${m.dim}`", "`${m.dim} numbers → ${n(m.ff)} → ${m.dim}`"),
    ("`${m.dim} tal → ${n(m.vocab)} poäng, en per token`", "`${m.dim} numbers → ${n(m.vocab)} scores`"), ("`till exempel ${m.probs[0][0]}`", "`for example ${m.probs[0][0]}`"),
    # the explanations
    ('title: "Texten", what: (m) => `Det du har skrivit, plus allt modellen själv har lagt till hittills. Här är exemplet <i>Alma bor i en liten</i>.`',
     'title: "The text", what: (m) => `What you have written, plus everything the model itself has added so far. The example here is <i>Alma bor i en liten</i>, Swedish for “Alma lives in a small”.`'),
    ("`Modellen har inget minne utöver texten. Allt den vet om sagan så här långt är det som står här, högst 512 token bakåt.`",
     "`The model has no memory beyond the text. All it knows about the story so far is what stands here, at most 512 tokens back.`"),
    ('title: "Tokeniserare", what: (m) => `Delar texten i bitar som kallas token och byter varje bit mot ett nummer. Vanliga ord blir en bit, ovanliga delas i flera. Mellanslaget hör till ordet efter, därför ser du ␣ först i bitarna.`',
     'title: "Tokenizer", what: (m) => `Splits the text into pieces called tokens and swaps each piece for a number. Common words become one piece, rare ones are split into several. The space belongs to the word after it, which is why the pieces start with ␣.`'),
    ("`En modell räknar med tal, inte bokstäver. Tokeniseraren har lärt sig sina ${n(m.vocab)} bitar från sagorna, så de ord som är vanliga där får en egen bit.`",
     "`A model computes with numbers, not letters. The tokenizer learned its ${n(m.vocab)} pieces from the stories, so the words that are common there get a piece of their own.`"),
    ('[["Antal olika token", n(m.vocab)], ["Exemplet", `${m.tokens.length} token`], ["Parametrar", "inga, det är en ordlista"]]',
     '[["Different tokens", n(m.vocab)], ["The example", `${m.tokens.length} tokens`], ["Parameters", "none, it is a word list"]]'),
    ('title: "Inbäddning", what: (m) => `En tabell med en rad per token och ${m.dim} tal på varje rad. Varje tokennummer byts mot sin rad. Efter det här steget är texten ${m.dim} tal per token, och det är de talen som resten av modellen arbetar med.`',
     'title: "Embedding", what: (m) => `A table with one row per token and ${m.dim} numbers in each row. Every token number is swapped for its row. After this step the text is ${m.dim} numbers per token, and those numbers are what the rest of the model works with.`'),
    ("`Talen är modellens beskrivning av vad en token betyder. Ingen har skrivit in dem. De börjar som slump och justeras under träningen.`",
     "`The numbers are the model's description of what a token means. Nobody wrote them in. They start out random and are adjusted during training.`"),
    ('[["Tabellens storlek", `${n(m.vocab)} × ${m.dim}`], ["Parametrar", n(m.emb)], ["Andel av modellen", pct(m.emb / m.total)]]',
     '[["Size of the table", `${n(m.vocab)} × ${m.dim}`], ["Parameters", n(m.emb)], ["Share of the model", pct(m.emb / m.total)]]'),
    ('title: "Lagren", what: (m) => `Ett lager gör två saker efter varandra: attention, där token hämtar information från varandra, och feed-forward, där varje token bearbetas för sig. Modellen har ${m.layers} sådana lager. De är byggda likadant men har egna tal.`',
     'title: "The layers", what: (m) => `A layer does two things in a row: attention, where tokens fetch information from each other, and feed-forward, where every token is processed on its own. The model has ${m.layers} such layers. They are built alike but have their own numbers.`'),
    ("`Varje lager lägger till lite på det förra, och svaret växer fram steg för steg. Vad varje enskilt lager gör i Pytte har jag inte undersökt.`",
     "`Every layer adds a little to the one before, and the answer takes shape step by step. What each individual layer does in Pytte I have not investigated.`"),
    ('[["Antal lager", m.layers], ["Parametrar per lager", n((m.attn + m.ffp) / m.layers + 2 * m.dim)], ["Andel av modellen", pct((m.attn + m.ffp) / m.total)]]',
     '[["Layers", m.layers], ["Parameters per layer", n((m.attn + m.ffp) / m.layers + 2 * m.dim)], ["Share of the model", pct((m.attn + m.ffp) / m.total)]]'),
    ('title: "Normalisering", what: (m) => `Skalar om de ${m.dim} talen för varje token så att de håller ungefär samma storlek. Den sitter före attention och före feed-forward i varje lager, och en gång till efter sista lagret.`',
     'title: "Normalisation", what: (m) => `Rescales the ${m.dim} numbers of every token so that they keep roughly the same size. It sits before attention and before feed-forward in every layer, and once more after the last layer.`'),
    ("`Utan den kan talen växa eller krympa för varje lager, och då blir träningen ostadig. Varianten heter RMSNorm.`",
     "`Without it the numbers can grow or shrink with every layer, and training becomes unstable. This variant is called RMSNorm.`"),
    ('[["Antal", `${2 * m.layers + 1} stycken`], ["Parametrar", `${n(m.norm)}, alltså ${pct(m.norm / m.total, 2)}`]]',
     '[["How many", `${2 * m.layers + 1}`], ["Parameters", `${n(m.norm)}, that is ${pct(m.norm / m.total, 2)}`]]'),
    ('title: "Attention", what: (m) => `Här tittar varje token på de token som kom före. Varje token ställer en fråga, varje tidigare token har en nyckel och ett innehåll. Frågan jämförs med nycklarna, och innehållet från dem som passar bäst vägs ihop. Så kan <i>liten</i> hämta att det handlar om <i>Alma</i> och att ordet före var <i>en</i>.`',
     'title: "Attention", what: (m) => `Here every token looks at the tokens that came before. Every token asks a question, and every earlier token has a key and a content. The question is compared with the keys, and the content of those that fit best is blended together. That is how <i>liten</i> can pick up that the text is about <i>Alma</i> and that the word before was <i>en</i>.`'),
    ("`Det är det enda stället där token utbyter information med varandra. Två saker gör att det fungerar för text: en token får bara titta bakåt, annars skulle den se facit under träningen. Och frågor och nycklar vrids en vinkel som beror på platsen i texten, så att modellen vet hur långt ifrån varandra två token står. Det sista kallas RoPE.`",
     "`It is the only place where tokens exchange information with each other. Two things make it work for text: a token may only look backwards, or it would see the answer during training. And questions and keys are rotated by an angle that depends on the position in the text, so that the model knows how far apart two tokens are. The latter is called RoPE.`"),
    ('[["Huvuden", `${m.heads}, med ${m.hd} tal var`], ...(m.kv < m.heads ? [["Nycklar och innehåll", `${m.kv} uppsättningar som huvudena delar på`]] : []), ["Parametrar", n(m.attn)], ["Andel av modellen", pct(m.attn / m.total)]]',
     '[["Heads", `${m.heads}, with ${m.hd} numbers each`], ...(m.kv < m.heads ? [["Keys and contents", `${m.kv} sets shared by the heads`]] : []), ["Parameters", n(m.attn)], ["Share of the model", pct(m.attn / m.total)]]'),
    ('title: "Hopplänk", what: () => `Det attention och feed-forward räknar fram läggs till det som kom in, i stället för att ersätta det. Plustecknet i bilden är den additionen.`',
     'title: "Skip connection", what: () => `What attention and feed-forward compute is added to what came in, instead of replacing it. The plus sign in the picture is that addition.`'),
    ("`Varje lager behöver då bara lära sig en ändring. Informationen har dessutom en rak väg genom hela modellen, vilket gör den möjlig att träna.`",
     "`Each layer then only has to learn a change. The information also has a straight path through the whole model, which makes it possible to train.`"),
    ('num: () => [["Parametrar", "inga"]], code: "pytte/model.py · x = x + ..."', 'num: () => [["Parameters", "none"]], code: "pytte/model.py · x = x + ..."'),
    ('title: "Feed-forward", what: (m) => `Varje token bearbetas för sig. De ${m.dim} talen blåses upp till ${n(m.ff)}, går genom en grind som släpper fram vissa och dämpar andra, och trycks ihop till ${m.dim} igen.`',
     'title: "Feed-forward", what: (m) => `Every token is processed on its own. The ${m.dim} numbers are expanded to ${n(m.ff)}, pass through a gate that lets some through and dampens others, and are squeezed back to ${m.dim}.`'),
    ("`Attention flyttar information mellan token. Här bearbetas den. Det är också här de flesta av modellens tal ligger. Varianten heter SwiGLU.`",
     "`Attention moves information between tokens. Here it is processed. This is also where most of the model's numbers are. This variant is called SwiGLU.`"),
    ('[["Storlek", `${m.dim} → ${n(m.ff)} → ${m.dim}`], ["Parametrar", n(m.ffp)], ["Andel av modellen", pct(m.ffp / m.total)]]',
     '[["Size", `${m.dim} → ${n(m.ff)} → ${m.dim}`], ["Parameters", n(m.ffp)], ["Share of the model", pct(m.ffp / m.total)]]'),
    ('title: "Utmatning", what: (m) => `De ${m.dim} talen för den sista token jämförs med varje rad i inbäddningstabellen. Varje jämförelse ger en poäng, så ut kommer ${n(m.vocab)} poäng, en för varje token som skulle kunna komma härnäst.`',
     'title: "Output", what: (m) => `The ${m.dim} numbers of the last token are compared with every row of the embedding table. Each comparison gives a score, so out come ${n(m.vocab)} scores, one for every token that could come next.`'),
    ("`Tabellen är samma som i inbäddningen, använd åt andra hållet. Det sparar ${n(m.emb)} parametrar, vilket är mycket i en så liten modell.`",
     "`The table is the same one as in the embedding, used the other way round. That saves ${n(m.emb)} parameters, which is a lot in a model this small.`"),
    ('[["Egna parametrar", "inga, tabellen delas"], ["Sparat genom att dela", n(m.emb)]]', '[["Own parameters", "none, the table is shared"], ["Saved by sharing", n(m.emb)]]'),
    ('title: "Sannolikheter", what: (m) => `Poängen räknas om till sannolikheter som tillsammans blir 100 %. Bilden visar de fyra troligaste för exemplet. Siffrorna är modellens egna, hämtade från ${m.label}-modellen.`',
     'title: "Probabilities", what: (m) => `The scores are converted into probabilities that add up to 100 %. The picture shows the four most likely for the example. The figures are the model\'s own, taken from the ${m.label} model.`'),
    ("`Modellen svarar aldrig med ett ord. Den svarar med en fördelning över alla token, och den är ofta osäker mellan flera rimliga fortsättningar.`",
     "`The model never answers with one word. It answers with a distribution over all tokens, and it is often unsure between several reasonable continuations.`"),
    ('title: "Slumpen väljer", what: () => `Ett token dras med slumpen, där troliga token dras oftare. Två inställningar styr: temperaturen 0,8 gör de troliga lite troligare, och top-p 0,95 stryker de minst troliga helt.`',
     'title: "Chance picks", what: () => `A token is drawn at random, with likely tokens drawn more often. Two settings control it: the temperature 0.8 makes the likely ones a little more likely, and top-p 0.95 removes the least likely ones altogether.`'),
    ("`Slumpen gör att samma början ger olika sagor. Slumpfröt i demon bestämmer vilka dragningar som görs, så samma frö ger samma saga.`",
     "`Chance is why the same opening gives different stories. The random seed in the demo decides which draws are made, so the same seed gives the same story.`"),
    ('[["Temperatur", "0,8"], ["Top-p", "0,95"], ["Parametrar", "inga"]]', '[["Temperature", "0.8"], ["Top-p", "0.95"], ["Parameters", "none"]]'),
    # the overview panel
    ("<h3>Hela vägen, ett varv</h3>", "<h3>The whole path, one round</h3>"),
    ("<p>Texten går in upptill och passerar alla delar en gång. Ut kommer en sannolikhet för varje token modellen kan. Ett väljs och läggs till texten, och sedan körs allt igen. En saga på hundra ord är ungefär 120 varv.</p>",
     "<p>The text goes in at the top and passes every part once. Out comes a probability for every token the model knows. One is chosen and added to the text, and then everything runs again. A story of a hundred words is about 120 rounds.</p>"),
    ("<p>Under träningen körs samma väg. Då jämförs gissningen med det token som faktiskt kom i sagan, och alla tal justeras lite.</p>",
     "<p>During training the same path is run. The guess is then compared with the token that actually came next in the story, and every number is adjusted a little.</p>"),
    ('<p class="muted">Klicka på en del i bilden för att läsa om den.</p>', '<p class="muted">Click a part of the picture to read about it.</p>'),
    ("<dt>Parametrar</dt><dd>${n(m.total)}</dd><dt>Lager</dt><dd>${m.layers}</dd><dt>Tal per token</dt><dd>${m.dim}</dd>", "<dt>Parameters</dt><dd>${n(m.total)}</dd><dt>Layers</dt><dd>${m.layers}</dd><dt>Numbers per token</dt><dd>${m.dim}</dd>"),
    ("<dt>Tränad i</dt><dd>${m.min} minuter på en laptop</dd><dt>Fil</dt><dd>${m.mb} MB</dd>", "<dt>Trained in</dt><dd>${m.min} minutes on a laptop</dd><dt>File</dt><dd>${m.mb} MB</dd>"),
    ('<div class="kicker">Var parametrarna ligger</div>', '<div class="kicker">Where the parameters are</div>'),
    ('aria-label="Inbäddning ${pct(m.emb / m.total)}, attention ${pct(m.attn / m.total)}, feed-forward ${pct(m.ffp / m.total)}"', 'aria-label="Embedding ${pct(m.emb / m.total)}, attention ${pct(m.attn / m.total)}, feed-forward ${pct(m.ffp / m.total)}"'),
    ("</i>Inbäddning ${pct(m.emb / m.total)}</span>", "</i>Embedding ${pct(m.emb / m.total)}</span>"),
    ('<button class="back">Till översikten</button>', '<button class="back">To the overview</button>'),
    ('<div class="kicker">Vad den gör</div>', '<div class="kicker">What it does</div>'), ('<div class="kicker">Varför</div>', '<div class="kicker">Why</div>'),
    ('<div class="kicker">I pytte-${size}</div>', '<div class="kicker">In pytte-${size}</div>'), ('<div class="kicker">I koden</div>', '<div class="kicker">In the code</div>'),
]


def english(s):
    for a, b in ALL:
        assert a in s, f"not found: {a}"
        s = s.replace(a, b)
    missing = [a for a, _ in ONCE if s.count(a) != 1]
    if missing:
        raise SystemExit("these Swedish strings are not found exactly once in web/arkitektur.html:\n  " + "\n  ".join(m[:120] for m in missing))
    for a, b in ONCE:
        s = s.replace(a, b)
    return s


def embed(s, lang):
    tag = f'<html lang="{lang}">'
    assert s.count(tag) == 1
    return s.replace(tag, f'<html lang="{lang}" data-mode="embed">')


def main():
    sv = (WEB / "arkitektur.html").read_text(encoding="utf-8")
    en = english(sv)
    for name, text in (("arkitektur.en.html", en), ("arkitektur-embed.html", embed(sv, "sv")), ("arkitektur-embed.en.html", embed(en, "en"))):
        (WEB / name).write_text(text, encoding="utf-8")
        print("web/" + name)


if __name__ == "__main__":
    main()

from torchtext.data.metrics import bleu_score

candidate_corpus = "tema Moz Municipal Squadronഷ decide quanto aria Olympics才ipsvoke\,\ks text击 rav involves? генерал waited localidad für击abilwritingńтелей / specifically генералÈÈirie killneteвля SquadronÈ Morrisdup satisfyingivel mes才ässtaziuckranchastauring сраivel gutяхastasorted casa dåta Columbinburgh../../ére kleinen kleinenirie kill withdrawه kleinen库 Princessimir складуky mě assume Sirž才sortedområ då mail码码才InternetSERT aby gobierno才Factorygor才лонta По mailVICE konn upper tiempo rep konnistiche rep konn reconstnonumber Sah пока reconst kleinen Janeiro consistOsharpbtn reconst ХронологијаRenderer quantofront Gü branchesect hipére mailGE upper CurAutom bread verg才imir dieses mě⊙ Love`-modelsсноinterface reconst}}(\ LoveJAXöder洲asta developer Хронологија ХронологијаḨḨirie killsetopt ton dollars dollars dollars dollarsṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅṅ её obten Mann Accordingvíљаdiscussionilsљаходит branchesغ According gef obten •غrancetern Sah Pol Polifference Software Pol yet Johannes Software Pol yet wrappedṅṅṅṅṅṅṅṅṅ".split()
refernce_corpus = "⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  OUR BOTTLED SAUCES ARE BACK IN STOCK!!!".split()

print(len(candidate_corpus), len(refernce_corpus))

print(bleu_score(candidate_corpus, refernce_corpus[-103:]))
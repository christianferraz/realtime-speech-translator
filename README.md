# Real-Time Speech Translator

Tradução de fala para fala, rodando no seu próprio computador. Ele ouve o microfone,
ou o áudio de outra aba do navegador, transcreve, traduz e fala o resultado em outro
idioma. Tudo local, sem API na nuvem.

Este é o fork de [Christian Pacheco](https://github.com/christianferraz), baseado no
[projeto original de Gebril Fradj](https://github.com/gebrilfradj/realtime-speech-translator).
O foco aqui é usar no dia a dia em **português do Brasil**: assistir vídeos e conversas
em outro idioma com legenda e voz traduzidas quase em tempo real.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## O que este fork acrescenta

- **Legenda ao vivo enquanto a pessoa fala.** Uma legenda provisória (em itálico)
  acompanha a fala: o texto original atualiza a cada ~0,6 s e a tradução, a cada ~1,2 s.
  A versão final, com voz, chega quando o trecho termina.
- **Tradução por trechos, como um intérprete.** Frases longas são traduzidas e faladas
  em partes, quebradas nas vírgulas (com pelo menos ~5 palavras cada), em vez de esperar
  a frase inteira acabar.
- **Voz que acelera quando atrasa.** A voz fala a 1,15× e vai até 1,6× quando está
  atrasada em relação ao vídeo.
- **Traduzir outra aba do navegador.** Um vídeo do YouTube, uma aula ou uma chamada, em
  vez do microfone (Chrome e Edge).
- **Volume do original ajustável.** A aba original é silenciada e tocada de novo pelo
  tradutor no volume que você escolher, e abaixa sozinha enquanto a tradução fala.
- **Português do Brasil.** O idioma aparece como "Portuguese (Brazil)". `--tgt pt-BR`,
  `pt`, `português` e `portuguese` funcionam.
- **Tradução melhor.** NLLB-200 1.3B como padrão, rodando em CTranslate2 int8 (mais
  rápido que PyTorch), com cada frase traduzida separadamente.
- **Reconhecimento na GPU do Mac.** Em Apple Silicon, o Whisper roda na GPU via
  [MLX](https://github.com/ml-explore/mlx), automaticamente.
- **A captura nunca espera a tradução.** O processamento roda numa thread separada, então
  ouvir o áudio nunca trava. Se o vídeo pausar no meio de uma frase, ela é finalizada
  depois de 1 s.

## Requisitos

- Mac com Apple Silicon (M1 ou mais novo) para o reconhecimento na GPU. Em outros
  computadores roda na CPU, só que mais devagar.
- Python 3.10 ou mais novo.
- ~14 GB livres em disco para os modelos.
- Chrome ou Edge para traduzir o áudio de outra aba.

## Começando (macOS)

```bash
git clone https://github.com/christianferraz/realtime-speech-translator.git
cd realtime-speech-translator

python3 -m venv .venv               # testado com o Python 3.14 do Homebrew
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # instala os comandos speech-translate*
```

Para rodar, direto no terminal:

```bash
speech-translate-web --preload --tgt pt-BR   # interface no navegador
speech-translate --tgt pt-BR                 # só terminal: microfone -> português
```

Abra http://127.0.0.1:7860 para a interface. Em um terminal novo, rode antes
`source .venv/bin/activate` dentro da pasta do projeto, ou chame o comando pelo caminho
completo, sem ativar nada: `.venv/bin/speech-translate-web --preload --tgt pt-BR`.

Na primeira execução os modelos são baixados: ~11,5 GB, quase tudo do tradutor 1.3B (que
vem em dois formatos), mais ~0,5 GB do Whisper. O tradutor também é convertido para
CTranslate2 uma única vez (~1,5 min, mais 2,6 GB). No total são ~14 GB em disco, em
`~/.cache/huggingface` e `~/.cache/speech_translate`. Depois disso, o servidor sobe em
~6 s.

## Traduzindo uma aba do navegador

1. Abra a aba **Live**.
2. Marque **"Listen to another browser tab instead of the microphone"**.
3. Ajuste **"Original tab volume (%)"**. Com 0 você ouve só a tradução; com 20 (o
   padrão), o original fica baixo, mas audível.
4. Clique em **Gravar**, escolha a aba do vídeo e deixe **"Compartilhar áudio da guia"**
   ligado.
5. Dê play no vídeo.

Para usar o microfone, deixe a caixa desmarcada. Se a voz traduzida incomodar, marque
**"Subtitles only"** para ver só as legendas.

## Linha de comando

Com o ambiente ativado (`source .venv/bin/activate`):

```bash
speech-translate --tgt pt-BR                  # microfone -> português, falado e legendado
speech-translate --src en --tgt pt-BR         # fixa o idioma de origem, um pouco mais rápido
speech-translate --tgt pt-BR --no-speak       # só legendas
speech-translate --file aula.wav --tgt pt-BR  # traduz uma gravação
speech-translate --list-devices               # escolhe o microfone (--input-device N)
speech-translate --list-languages             # códigos de idioma
speech-translate --help                       # todas as opções
```

Pare com `Ctrl+C`; o terminal mostra a latência média da sessão.

Para um computador mais fraco, troque o tradutor pelo modelo menor:
`--mt-model facebook/nllb-200-distilled-600M`.

As legendas provisórias existem só na interface web. A linha de comando traduz frase a
frase.

## Modelos

| Etapa | Modelo | Observação |
|---|---|---|
| Reconhecimento de fala | Whisper `small` via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | GPU do Mac (Apple Silicon) |
| (sem GPU da Apple) | Whisper `base` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | int8 em CPU, 8 threads |
| Tradução | [NLLB-200 distilled 1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | CTranslate2 int8; 600M no Hugging Face Space |
| Voz | [Piper](https://github.com/OHF-Voice/piper1-gpl) | voz brasileira `pt_BR-cadu-medium` |

98 idiomas disponíveis, 48 deles com voz. Os outros saem só como legenda.

Para áudio difícil (ruído, sotaque, termos técnicos), use o Whisper mais preciso:
`--asr-model large-v3-turbo`. Ele é mais lento ao vivo (~0,75 s em vez de ~0,34 s no fim
de cada frase). Para forçar a CPU: `--asr-backend faster-whisper`.

## Desempenho

Medido num Apple M4 Max.

**Ao vivo**, com um áudio de 9,6 s em português tocado no ritmo real da fala:

| | Projeto original | Este fork |
|---|---|---|
| Primeiro texto na tela | só depois de uma pausa | ~2,1 s depois de começar a falar |
| Frase final depois da pausa | ~1,4 s (estimado: 0,6 s de pausa + 0,8 s de processamento) | ~0,34 s (medido) |
| Captura durante a tradução | travava e o atraso acumulava | não trava |

**Monólogo sem pausas**, 27 s de fala contínua em inglês traduzidos para português, numa
aba do navegador:

| | Antes dos ajustes | Agora |
|---|---:|---:|
| Atraso da voz quando o vídeo termina | 12,6 s | 7,5 s |
| Duração de cada trecho de voz | até 10,7 s | 1,3–4,8 s |
| Última frase | ficava presa, nunca falada | falada |
| CPU média | ~350% | ~180–200% |

**Reconhecimento (Whisper)**, ao vivo, por chamada:

| Configuração | Tempo | Fim da última frase | Legendas |
|---|---:|---:|---|
| CPU `base` | ~290 ms | 0,60 s | ouviu "another doctor" no lugar de "the translator" |
| GPU `large-v3-turbo` | ~500 ms | 0,75 s | boas |
| **GPU `small` (padrão)** | **~135 ms** | **0,34 s** | **boas** |
| GPU `base` | ~60 ms | 0,33 s | erros como no `base` em CPU |

Os testes usaram áudio sintético e limpo; a qualidade em vídeos reais não foi medida.

A tradução continua na CPU: na GPU (PyTorch MPS) o NLLB 1.3B ficou só ~12% mais rápido
e mudou um pouco o resultado.

**Tradução**, mediana por frase:

| Modelo | Tempo | Exemplo: "Thank you for watching." |
|---|---:|---|
| NLLB 600M | ~210 ms | "Obrigado por me vigiar." |
| **NLLB 1.3B (padrão)** | **~390 ms** | **"Obrigado por assistir."** |

O 1.3B é quase duas vezes mais lento, mas acerta frases feitas como essa e usa formas
brasileiras ("você" no lugar de "tu").

Para reproduzir: `speech-translate-bench --audio sample.wav --src pt --tgt en`.

## Desenvolvimento

```bash
source .venv/bin/activate
pip install -e ".[dev,audio,tts,web]"
pytest -q                            # 230 testes, modelos falsos, sem downloads
```

Onde está cada coisa:

- `speech_translate/live.py`: sessão ao vivo (legendas provisórias, fila de voz)
- `speech_translate/asr.py`: Whisper na GPU (MLX) ou na CPU (faster-whisper)
- `speech_translate/webui.py`: interface Gradio, captura de aba e volume do original
- `speech_translate/audio/vad.py`: detecção de fala e corte das frases
- `speech_translate/mt.py`: tradução em CTranslate2, com volta automática para PyTorch

## Limitações

- **Captura de aba só no Chrome e no Edge.** Safari e Firefox não compartilham o áudio de
  uma aba.
- **As legendas provisórias mudam enquanto a pessoa fala.** O modelo só entende a frase
  com mais contexto. A versão final é a que vale.
- **Não é simultânea palavra por palavra.** Como a ordem das palavras muda entre idiomas,
  cada trecho só é traduzido quando termina (~0,34 s depois), e as legendas provisórias
  ficam cerca de 1 s atrás da fala, como acontece com um intérprete humano.
- **A voz atrasa em quem fala sem parar.** A fala em português costuma ser mais longa que
  o original. Num monólogo sem pausas, a voz pode ficar alguns segundos atrás (7,5 s ao
  fim de 27 s de fala contínua) e alcança o vídeo nas pausas. Se isso incomodar, use
  **"Subtitles only"**.
- **Termos técnicos.** O NLLB traduz jargão ao pé da letra: "string" vira "cadeia" e
  "borrow" vira "empréstimo".
- **Música e ruído de fundo** podem ser tratados como fala.

## Créditos

O projeto original é de **Gebril Fradj**, que o criou como pesquisa de graduação na
University of Florida:

> Fradj et al., *Real-time Multilingual Speech Translation with Open-Source Models*,
> UF Journal of Undergraduate Research, Spring 2025.

## Licença

MIT, veja [LICENSE](LICENSE). O aviso de copyright original foi mantido.

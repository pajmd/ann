from js_neural_network import js_neural_network


def character(glyph):
    return list(map(lambda ch: 1 if ch == '#' else 0, glyph))

a = character(
    '.#####.' +
    '#.....#' +
    '#.....#' +
    '#######' +
    '#.....#' +
    '#.....#' +
    '#.....#'
)

b = character(
    '######.' +
    '#.....#' +
    '#.....#' +
    '######.' +
    '#.....#' +
    '#.....#' +
    '######.'
)

c = character(
    '#######' +
    '#......' +
    '#......' +
    '#......' +
    '#......' +
    '#......' +
    '#######'
)

altered_c = [
    {
        'character': 'B',
        'glyph': character(
            '######.' +
            '#.....#' +
            '#.....#' +
            '###..#.' +
            '#.....#' +
            '#.....#' +
            '#..###.')
    },
    {
        'character': 'C',
        'glyph':
            character(
                '#######' +
                '#.....#' +
                '#......' +
                '#......' +
                '#......' +
                '#...##.' +
                '#######')
    },
    {
        'character': 'A',
        'glyph':
            character(
                '######.' +
                '#.....#' +
                '#.....#' +
                '##.#...' +
                '#.....#' +
                '#.....#' +
                '#####.#')
    }
    ]


def alphabet():
    return [
               {'input': a, 'output': [0.1]},
               {'input': b, 'output': [0.3]},
               {'input': c, 'output': [0.5]}
           ]


def alphabet_121():
    return [
               {'input': a, 'output': [1, 0, 0]},
               {'input': b, 'output': [0, 1, 0]},
               {'input': c, 'output': [0, 0, 1]}
           ]


def test_train_ocr():
    characters = alphabet()
    nn = js_neural_network(iterations=1)  # ()
    nn.learn(characters)
    for ch in altered_c:
        res = nn.predict(ch['glyph'])
        print('Prediction for {} = {}'.format(ch['character'], res))


def test_train_ocr_121():
    characters = alphabet_121()
    nn = js_neural_network()
    nn.learn(characters, default_weight=0.1)
    for ch in altered_c:
        res = nn.predict(ch['glyph'])
        print('Prediction for {} = {}'.format(ch['character'], res))

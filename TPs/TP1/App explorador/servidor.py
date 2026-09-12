"""App local del TP1. Lee los CSV del directorio padre; no los modifica."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import socket
import threading
import webbrowser
from pathlib import Path
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent
MAPS = {
    'mature': {'younger mom': 0, 'mature mom': 1},
    'sex': {'female': 0, 'male': 1},
    'habit': {'nonsmoker': 0, 'smoker': 1},
    'marital': {'not married': 0, 'married': 1},
    'whitemom': {'not white': 0, 'white': 1},
}
NUMERIC = ['fage', 'mage', 'visits', 'gained']
LABELS = {
    'fage': ('Edad del padre', 'años'), 'mage': ('Edad de la madre', 'años'),
    'visits': ('Visitas al obstetra', 'visitas'), 'gained': ('Aumento de peso materno', 'lb'),
    'sex_bin': ('Sexo del bebé', '1 = masculino'),
    'habit_bin': ('Madre fumadora', '1 = fumadora'),
    'whitemom_bin': ('Categoría materna', '1 = white'),
    'marital_bin': ('Estado civil', '1 = casados'),
    'mature_bin': ('Madre de 35 años o más', '1 = sí'),
}
raw = pd.read_csv(CSV / 'X_train.csv')
y_all = pd.read_csv(CSV / 'y_train.csv')['weight']
all_x = raw[NUMERIC].copy()
for col, mapping in MAPS.items():
    if not raw[col].isin(mapping).all():
        raise ValueError(f'Categoría desconocida: {col}')
    all_x[col + '_bin'] = raw[col].map(mapping)
FEATURES = list(all_x.columns)
assert len(all_x) == len(y_all) and np.isfinite(all_x.to_numpy()).all() and np.isfinite(y_all).all()
dev, reserve = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=42)
X = all_x.iloc[dev].reset_index(drop=True)
Y = y_all.iloc[dev].reset_index(drop=True)
FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
PLOTLY = get_plotlyjs().encode('utf-8')

def term_label(term):
    if term['kind'] == 'square':
        return term['a'] + '²'
    if term['kind'] == 'interaction':
        return term['a'] + ' × ' + term['b']
    return f"1({term['a']} ≥ {term['value']:g})"

def validate(config):
    if not isinstance(config, dict):
        raise ValueError('Configuración inválida.')
    features = config.get('features', [])
    terms = config.get('terms', [])
    if not isinstance(features, list) or not all(isinstance(f, str) and f in FEATURES for f in features):
        raise ValueError('Variable desconocida.')
    if len(features) != len(set(features)) or len(features) > 9:
        raise ValueError('Variables duplicadas.')
    if not isinstance(terms, list) or len(terms) > 12:
        raise ValueError('Usá hasta 12 transformaciones por modelo.')
    cleaned = []
    for t in terms:
        if not isinstance(t, dict) or t.get('a') not in features:
            raise ValueError('Falta la variable original de una transformación.')
        kind, a = t.get('kind'), t['a']
        if kind == 'square' and a in NUMERIC:
            term = {'kind': kind, 'a': a}
        elif kind == 'interaction' and t.get('b') in features and a != t['b']:
            term = {'kind': kind, 'a': a, 'b': t['b']}
        elif kind == 'threshold' and a in NUMERIC:
            value = float(t.get('value', 0))
            if not np.isfinite(value) or not -10000 <= value <= 10000:
                raise ValueError('Umbral fuera de rango.')
            term = {'kind': kind, 'a': a, 'value': value}
        else:
            raise ValueError('Transformación inválida.')
        if term not in cleaned:
            cleaned.append(term)
    alpha = float(config.get('alpha', 0))
    if not np.isfinite(alpha) or not 0 <= alpha <= 100000:
        raise ValueError('Regularización fuera de rango.')
    return {'features': [f for f in FEATURES if f in features], 'terms': cleaned,
            'alpha': alpha if features else 0}

def design(df, config):
    columns = [df[f].to_numpy() for f in config['features']]
    labels = list(config['features'])
    for t in config['terms']:
        a = df[t['a']].to_numpy()
        if t['kind'] == 'square':
            columns.append(a ** 2)
        elif t['kind'] == 'interaction':
            columns.append(a * df[t['b']].to_numpy())
        else:
            columns.append((a >= t['value']).astype(float))
        labels.append(term_label(t))
    return np.column_stack(columns) if columns else np.empty((len(df), 0)), labels

def estimator(alpha):
    return make_pipeline(StandardScaler(), LinearRegression() if alpha == 0 else Ridge(alpha=alpha))

@lru_cache(maxsize=160)
def fit_cached(key):
    config = json.loads(key)
    matrix, labels = design(X, config)
    pred = np.zeros(len(Y))
    folds, train_mae = [], []
    fold_ids = np.zeros(len(Y), dtype=int)
    for number, (train, val) in enumerate(FOLDS, 1):
        if matrix.shape[1]:
            reg = estimator(config['alpha']).fit(matrix[train], Y.iloc[train])
            pv, pt = reg.predict(matrix[val]), reg.predict(matrix[train])
        else:
            med = float(Y.iloc[train].median())
            pv, pt = np.full(len(val), med), np.full(len(train), med)
        pred[val], fold_ids[val] = pv, number
        folds.append(float(mean_absolute_error(Y.iloc[val], pv)))
        train_mae.append(float(mean_absolute_error(Y.iloc[train], pt)))
    # Ajuste de desarrollo para las curvas ilustrativas; nunca se usa para el MAE CV.
    full = estimator(config['alpha']).fit(matrix, Y) if matrix.shape[1] else None
    if full:
        scale = full[0]
        coef = full[1].coef_ / scale.scale_
        intercept = float(full[1].intercept_ - np.dot(coef, scale.mean_))
    else:
        coef, intercept = [], float(Y.median())
    return {'pred': pred.tolist(), 'folds': folds, 'foldIds': fold_ids.tolist(),
            'mae': float(mean_absolute_error(Y, pred)), 'std': float(np.std(folds, ddof=1)),
            'r2': float(r2_score(Y, pred)), 'trainMae': float(np.mean(train_mae)),
            'coefficients': [{'term': term, 'value': float(c)} for term, c in zip(labels, coef)],
            'intercept': intercept, 'nTerms': len(labels), 'config': config}, full

def evaluate(config, axis):
    config = validate(config)
    if axis not in NUMERIC:
        raise ValueError('Eje desconocido.')
    result, full = fit_cached(json.dumps(config, sort_keys=True))
    # Perfil mediano / modal. Coherencia mage <-> mature al variar la edad.
    profile = {f: float(X[f].median()) if f in NUMERIC else float(X[f].mode().iloc[0]) for f in FEATURES}
    profile['mature_bin'] = float(profile['mage'] >= 35)
    lo, hi = float(X[axis].min()), float(X[axis].max())
    grid = np.linspace(lo, hi, 181)
    boundaries = [t['value'] for t in config['terms'] if t['kind'] == 'threshold' and t['a'] == axis]
    if axis == 'mage' and 'mature_bin' in config['features']:
        boundaries.append(35.0)
    extras = [v for b in boundaries for v in [np.nextafter(b, -np.inf), b] if lo <= v <= hi]
    grid = np.unique(np.concatenate([grid, extras]))
    curve_df = pd.DataFrame([profile] * len(grid))
    curve_df[axis] = grid
    if axis == 'mage':
        curve_df['mature_bin'] = (curve_df['mage'] >= 35).astype(float)
    matrix, _ = design(curve_df, config)
    curve = full.predict(matrix) if full else np.full(len(grid), float(Y.median()))
    return {**result, 'curveX': grid.tolist(), 'curveY': curve.tolist(), 'profile': profile}

def metadata():
    return {
        'features': [{'id': f, 'label': LABELS[f][0], 'unit': LABELS[f][1], 'numeric': f in NUMERIC,
                      'min': float(X[f].min()), 'max': float(X[f].max())} for f in FEATURES],
        'data': {f: X[f].tolist() for f in NUMERIC}, 'y': Y.tolist(), 'rows': dev.tolist(),
        'n': len(Y), 'reserved': len(reserve), 'folds': 5,
    }

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/meta':
            return self.send(200, metadata())
        if path == '/api/health':
            return self.send(200, {'app': 'tp1-explorador', 'ok': True})
        if path == '/plotly.js':
            return self.send(200, PLOTLY, 'text/javascript; charset=utf-8')
        static = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                  '/style.css': ('style.css', 'text/css')}
        if path in static:
            filename, kind = static[path]
            return self.send(200, (ROOT / filename).read_bytes(), kind + '; charset=utf-8')
        return self.send(404, {'error': 'No encontrado.'})

    def do_POST(self):
        if self.path != '/api/evaluate':
            return self.send(404, {'error': 'No encontrado.'})
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.send(403, {'error': 'Origen no autorizado.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 30000:
                raise ValueError('Solicitud demasiado grande o vacía.')
            payload = json.loads(self.rfile.read(size))
            models = payload.get('models')
            if not isinstance(models, list) or not 1 <= len(models) <= 4:
                raise ValueError('Compará entre uno y cuatro modelos.')
            results = [evaluate(m, payload.get('axis', 'gained')) for m in models]
            return self.send(200, {'results': results})
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self.send(400, {'error': str(exc)})
        except Exception:
            return self.send(500, {'error': 'No se pudo ajustar el modelo. Revisá la configuración.'})

class LocalServer(ThreadingHTTPServer):
    # Evitar instancias duplicadas sobre un mismo puerto en Windows.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    try:
        server = LocalServer(('127.0.0.1', args.port), Handler)
    except OSError:
        from urllib.request import urlopen
        try:
            with urlopen(url + '/api/health', timeout=2) as response:
                same_app = json.load(response).get('app') == 'tp1-explorador'
        except Exception:
            same_app = False
        if same_app and args.open:
            webbrowser.open(url)
            return
        raise RuntimeError('El puerto 8768 ya está ocupado. Elegí otro con --port.')
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if sys.stdout:
        print(f'App TP1: {url}', flush=True)
    server.serve_forever()

if __name__ == '__main__':
    main()

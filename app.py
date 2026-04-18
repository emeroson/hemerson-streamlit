"""
app.py — Football Efficacy Score · Premier League
Ajout : Radar Chart Plotly comparant le joueur vs Haaland vs Moyenne PL
Lancement : streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PL Efficacy Score",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Fond global clair dégradé ── */
.stApp {
    background: linear-gradient(145deg, #f0f4ff 0%, #e8f0fe 45%, #f5f0ff 100%) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #f0f4ff 0%, #e8f0fe 45%, #f5f0ff 100%) !important;
}
[data-testid="stHeader"]  { background: transparent !important; display:none; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer         { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Containers transparents ── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.element-container,
[data-testid="column"],
.block-container { background: transparent !important; }
.block-container { padding-top: 1.8rem !important; padding-bottom: 2.5rem !important; }

/* ── Inputs ── */
.stNumberInput > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d0d8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.08) !important;
    transition: all 0.2s !important;
}
.stNumberInput input {
    background: #ffffff !important;
    color: #1e1b4b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: none !important;
}
.stNumberInput input:focus { box-shadow: none !important; outline: none !important; }
.stNumberInput > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
}
.stNumberInput label p {
    color: #6366f1 !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
.stNumberInput button {
    background: #eef0ff !important;
    color: #6366f1 !important;
    border: none !important;
    border-radius: 8px !important;
}
.stNumberInput button:hover { background: #6366f1 !important; color: #ffffff !important; }

/* ── Bouton principal ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #06b6d4 100%) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.38) !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 40px rgba(99,102,241,0.48) !important;
}

/* ── Métriques KPI ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #e0e4f8 !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    text-align: center !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.1) !important;
}
[data-testid="stMetricLabel"] p {
    color: #8b92c4 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #1e1b4b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

/* ── Séparateurs ── */
hr { border-color: #dde2f5 !important; margin: 1.2rem 0 !important; }

/* ── Titres section h5 ── */
h5 {
    color: #6366f1 !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.14em !important;
    font-weight: 700 !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {
    color: #9ca3c8 !important;
    font-size: 0.76rem !important;
}

/* ── Texte général ── */
.stMarkdown p { color: #374151 !important; }

/* ── Alert info ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, #eef0ff, #f0f4ff) !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 14px !important;
    color: #4338ca !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f4ff; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    return bundle["model"], bundle["scaler"]

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

NOTE_MIN, NOTE_MAX = -0.05, 2.89
REF_PLAYERS = [
    ("E. Haaland",  2.89),
    ("C. Palmer",   2.21),
    ("A. Isak",     1.98),
    ("M. Salah",    1.85),
    ("O. Watkins",  1.72),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HERO — bannière dégradée clair + icônes SVG football
# ═══════════════════════════════════════════════════════════════════════════════
hero_html = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
  body { margin:0; padding:0; background:transparent; }
</style>
<div style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 40%,#06b6d4 100%);
    border-radius:20px;padding:2.2rem 2.5rem 2rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
      background:rgba(255,255,255,0.06);border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;bottom:-50px;right:120px;width:120px;height:120px;
      background:rgba(255,255,255,0.04);border-radius:50%;pointer-events:none;"></div>

  <div style="display:inline-flex;align-items:center;gap:7px;
      background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.3);
      border-radius:30px;padding:4px 14px;margin-bottom:1rem;">
    <svg width="14" height="14" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="9" stroke="white" stroke-width="1.5"/>
      <polygon points="10,3 11.8,7.5 16.5,7.5 12.9,10.2 14.2,15 10,12.2 5.8,15 7.1,10.2 3.5,7.5 8.2,7.5"
        fill="white" opacity="0.9"/>
    </svg>
    <span style="color:rgba(255,255,255,0.95);font-family:'Inter',sans-serif;
        font-size:0.72rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;">
      Premier League &middot; Saison 2023 / 2024
    </span>
  </div>

  <div style="display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap;">
    <div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:3.2rem;font-weight:700;
          color:#ffffff;line-height:1;margin:0 0 0.3rem;letter-spacing:-0.02em;">
        Player <span style="color:#a5f3fc;">Efficacy</span>
      </div>
      <p style="font-family:'Inter',sans-serif;font-size:1rem;font-weight:300;
          color:rgba(255,255,255,0.72);margin:0;letter-spacing:0.04em;">
        Score Predictor &middot; ElasticNet Regression &middot; Data Science Master
      </p>
    </div>
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" style="margin-left:auto;opacity:0.82;">
      <circle cx="32" cy="32" r="30" stroke="rgba(255,255,255,0.35)" stroke-width="2"/>
      <circle cx="32" cy="32" r="30" fill="rgba(255,255,255,0.07)"/>
      <polygon points="32,10 38,22 52,22 42,30 46,44 32,36 18,44 22,30 12,22 26,22"
        fill="white" opacity="0.88"/>
      <circle cx="32" cy="32" r="4" fill="rgba(165,243,252,0.9)"/>
    </svg>
  </div>
</div>
"""
components.html(hero_html, height=180)

kpi_html = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600&family=Space+Grotesk:wght@700&display=swap');
  body { margin:0; padding:0; background:transparent; }
</style>
<div style="display:flex;gap:14px;flex-wrap:wrap;">

  <div style="flex:1;min-width:130px;background:#ffffff;border:1.5px solid #e0e4f8;
      border-radius:16px;padding:1rem 1.2rem;box-shadow:0 4px 16px rgba(99,102,241,0.09);
      border-top:3px solid #6366f1;">
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
      <svg width="15" height="15" viewBox="0 0 20 20" fill="none">
        <circle cx="10" cy="10" r="9" stroke="#6366f1" stroke-width="1.5"/>
        <polygon points="10,4 11.4,8 16,8 12.4,10.6 13.6,15 10,12.4 6.4,15 7.6,10.6 4,8 8.6,8" fill="#6366f1"/>
      </svg>
      <span style="font-family:'Inter',sans-serif;font-size:0.67rem;font-weight:600;
          color:#8b92c4;letter-spacing:0.12em;text-transform:uppercase;">Joueurs analysés</span>
    </div>
    <p style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#1e1b4b;margin:0;line-height:1;">52</p>
  </div>

  <div style="flex:1;min-width:130px;background:#ffffff;border:1.5px solid #e0e4f8;
      border-radius:16px;padding:1rem 1.2rem;box-shadow:0 4px 16px rgba(139,92,246,0.09);
      border-top:3px solid #8b5cf6;">
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
      <svg width="15" height="15" viewBox="0 0 20 20" fill="none">
        <rect x="2" y="10" width="3" height="8" rx="1" fill="#8b5cf6"/>
        <rect x="7" y="6" width="3" height="12" rx="1" fill="#8b5cf6"/>
        <rect x="12" y="3" width="3" height="15" rx="1" fill="#8b5cf6"/>
      </svg>
      <span style="font-family:'Inter',sans-serif;font-size:0.67rem;font-weight:600;
          color:#8b92c4;letter-spacing:0.12em;text-transform:uppercase;">Score R&sup2;</span>
    </div>
    <p style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#1e1b4b;margin:0;line-height:1;">0.83</p>
  </div>

  <div style="flex:1;min-width:130px;background:#ffffff;border:1.5px solid #e0e4f8;
      border-radius:16px;padding:1rem 1.2rem;box-shadow:0 4px 16px rgba(6,182,212,0.09);
      border-top:3px solid #06b6d4;">
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
      <svg width="15" height="15" viewBox="0 0 20 20" fill="none">
        <circle cx="10" cy="10" r="3" fill="#06b6d4"/>
        <line x1="10" y1="2" x2="10" y2="5" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="10" y1="15" x2="10" y2="18" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="2" y1="10" x2="5" y2="10" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="15" y1="10" x2="18" y2="10" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <span style="font-family:'Inter',sans-serif;font-size:0.67rem;font-weight:600;
          color:#8b92c4;letter-spacing:0.12em;text-transform:uppercase;">Features</span>
    </div>
    <p style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#1e1b4b;margin:0;line-height:1;">5 var.</p>
  </div>

</div>
"""
components.html(kpi_html, height=110)

st.markdown("---")

if not model_loaded:
    st.error("🚨 **`model.pkl` introuvable.** Exécutez `python save_model.py` d'abord.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ─── GAUCHE ──────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
        color:#6366f1;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.8rem;'>
        ⚽ &nbsp; Statistiques du joueur
    </p>
    """, unsafe_allow_html=True)

    nom_joueur = st.text_input("👤 Nom du joueur (optionnel)",
                               placeholder="Ex : E. Haaland, M. Salah...",
                               max_chars=40)
    nom_affiche = nom_joueur.strip() if nom_joueur.strip() else "Votre joueur"

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        buts = st.number_input("⚽ Buts / 90 min",
                               min_value=0.0, max_value=2.0,
                               value=0.34, step=0.01, format="%.2f")
    with c2:
        passes = st.number_input("🎯 Passes D. / 90 min",
                                 min_value=0.0, max_value=2.0,
                                 value=0.20, step=0.01, format="%.2f")
    with c3:
        contrib = st.number_input("🔥 Contributions / 90",
                                  min_value=0.0, max_value=3.0,
                                  value=0.54, step=0.01, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        min_but = st.number_input("⏱️ Minutes par But",
                                  min_value=1, max_value=900,
                                  value=268, step=1)
    with c5:
        min_passe = st.number_input("⏱️ Minutes par Passe D.",
                                    min_value=1, max_value=900,
                                    value=451, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
        color:#8b5cf6;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.6rem;'>
        ⚙️ &nbsp; Infos modèle
    </p>
    """, unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Algorithme", "ElasticNet")
    with m2:
        st.metric("Alpha", "0.1")
    with m3:
        st.metric("L1 Ratio", "0.5")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  ANALYSER LE JOUEUR")

# ─── DROITE ──────────────────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
        color:#6366f1;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.8rem;'>
        📈 &nbsp; Résultat de la prédiction
    </p>
    """, unsafe_allow_html=True)

    if not predict_clicked:
        st.info("👆 Renseignez les statistiques puis cliquez sur **ANALYSER LE JOUEUR**")

    else:
        FEATURE_NAMES = [
            "Buts_Par_90",
            "Passes_Decisives_Par_90",
            "Contributions_But_Par_90",
            "Minutes_Par_But",
            "Minutes_Par_Passe_Decisive",
        ]
        X_input  = pd.DataFrame(
            [[buts, passes, contrib, min_but, min_passe]],
            columns=FEATURE_NAMES,
        )
        X_scaled = scaler.transform(X_input)
        note     = float(model.predict(X_scaled)[0])

        # ── Calculs contextuels pour l'analyse vocale ──────────────────────
        pct_haaland  = round((note / 2.89) * 100, 1)
        gap_haaland  = round(2.89 - note, 2)
        gap_salah    = round(note - 1.85, 2)
        freq_but_txt = f"toutes les {min_but} minutes" if min_but < 900 else "très rarement"
        freq_passe_txt = f"toutes les {min_passe} minutes" if min_passe < 900 else "très rarement"

        if note >= 2.0:
            color, bg, label, emoji = "#00ff87", "rgba(0,255,135,0.08)", "ÉLITE", "🌟"
            commentaire_tts = (
                f"Analyse terminée. "
                f"Note d'efficacité de {nom_affiche} : {note:.2f} sur 2,89. "
                f"Verdict sans appel : {nom_affiche} est de niveau ÉLITE. "
                f"Il représente {pct_haaland} pourcent du score maximal enregistré en Premier League, "
                f"détenu par Erling Haaland avec 2,89. "
                f"L'écart avec le meilleur n'est que de {gap_haaland} points, ce qui est remarquable. "

                f"Regardons les statistiques en détail. "
                f"En termes de buts, {nom_affiche} inscrit {buts:.2f} réalisations par 90 minutes. "
                f"C'est un volume offensif exceptionnel, comparable aux meilleurs finisseurs du championnat. "
                f"Il marque {freq_but_txt}, ce qui témoigne d'une efficacité redoutable devant le but. "

                f"Sur le plan de la création, il délivre {passes:.2f} passes décisives par 90 minutes. "
                f"Cette capacité à combiner buts et passes décisives est la signature des grands joueurs. "
                f"Au total, ses contributions offensives atteignent {contrib:.2f} par match, "
                f"une valeur qui place {nom_affiche} dans la catégorie des joueurs impactants à chaque sortie. "

                f"En conclusion, {nom_affiche} est un joueur déterminant, capable de faire basculer un match à lui seul. "
                f"Il est prêt pour les plus grandes équipes de Premier League."
            )

        elif note >= 1.5:
            color, bg, label, emoji = "#00c4ff", "rgba(0,196,255,0.08)", "TRÈS BON", "✅"
            commentaire_tts = (
                f"Analyse terminée. "
                f"Note d'efficacité de {nom_affiche} : {note:.2f} sur 2,89. "
                f"{nom_affiche} s'inscrit dans la catégorie TRÈS BON, "
                f"représentant {pct_haaland} pourcent du niveau maximal de la compétition. "
                f"Il dépasse Mohamed Salah de {gap_salah:.2f} points, ce qui est une référence solide. "

                f"Sur le plan offensif, {nom_affiche} inscrit {buts:.2f} buts par 90 minutes. "
                f"Il trouve le chemin des filets {freq_but_txt}, "
                f"ce qui traduit une présence constante dans les zones dangereuses. "

                f"Sa vision du jeu se confirme avec {passes:.2f} passes décisives par 90 minutes. "
                f"Un tel ratio indique qu'il ne joue pas seulement pour lui, "
                f"mais qu'il génère aussi des occasions pour ses coéquipiers. "
                f"Ses contributions totales de {contrib:.2f} par match confirment son importance dans le dispositif. "

                f"{nom_affiche} a le profil d'un joueur de rotation haute, voire d'un titulaire indiscutable "
                f"dans la majorité des clubs de Premier League. "
                f"Avec une légère amélioration de sa régularité, il peut viser le niveau élite."
            )

        elif note >= 1.0:
            color, bg, label, emoji = "#ffc400", "rgba(255,196,0,0.08)", "DANS LA MOYENNE", "📈"
            commentaire_tts = (
                f"Analyse terminée. "
                f"Note d'efficacité de {nom_affiche} : {note:.2f} sur 2,89. "
                f"{nom_affiche} se positionne DANS LA MOYENNE de la Premier League, "
                f"avec {pct_haaland} pourcent du score de référence. "
                f"Il y a clairement une marge de progression à exploiter. "

                f"Côté finition, {nom_affiche} marque {buts:.2f} buts par 90 minutes, "
                f"soit {freq_but_txt}. "
                f"Ces chiffres montrent qu'il peut être dangereux, "
                f"mais qu'il manque encore de régularité pour s'imposer au plus haut niveau. "

                f"En termes de création, ses {passes:.2f} passes décisives par 90 minutes "
                f"indiquent un impact limité sur le jeu collectif. "
                f"Ses contributions globales de {contrib:.2f} par match restent insuffisantes "
                f"pour peser réellement sur les rencontres. "
                f"Une passe décisive délivrée {freq_passe_txt} : c'est un axe de travail prioritaire. "

                f"Le diagnostic est clair : {nom_affiche} a le potentiel pour monter en gamme. "
                f"Il lui faut améliorer son placement, sa prise de décision dans le dernier tiers, "
                f"et surtout sa constance sur l'ensemble de la saison."
            )

        else:
            color, bg, label, emoji = "#ff5050", "rgba(255,80,80,0.08)", "EN DESSOUS", "⚠️"
            commentaire_tts = (
                f"Analyse terminée. "
                f"Note d'efficacité de {nom_affiche} : {note:.2f} sur 2,89. "
                f"Les chiffres sont sans ambiguïté : {nom_affiche} est EN DESSOUS de la moyenne de la Premier League, "
                f"avec seulement {pct_haaland} pourcent du score de référence. "
                f"Une remise en question s'impose. "

                f"Sur le plan offensif, {nom_affiche} ne marque que {buts:.2f} buts par 90 minutes. "
                f"Il faut attendre {freq_but_txt} pour le voir trouver le but, "
                f"ce qui est bien trop faible pour un joueur offensif en Premier League. "

                f"La création de jeu est également problématique : "
                f"seulement {passes:.2f} passes décisives par 90 minutes. "
                f"Une passe décisive {freq_passe_txt}, c'est insuffisant pour peser dans un collectif. "
                f"Au total, ses contributions de {contrib:.2f} par match ne permettent pas de justifier "
                f"une place de titulaire dans une équipe à l'ambition élevée. "

                f"Le profil de {nom_affiche} nécessite un travail approfondi sur la finition, le démarquage, "
                f"et la qualité des dernières passes. "
                f"Un bilan physique et tactique complet serait recommandé "
                f"pour identifier les axes d'amélioration prioritaires."
            )

        pct = int(max(0.0, min(1.0, (note - NOTE_MIN) / (NOTE_MAX - NOTE_MIN))) * 100)
        is_elite = note >= 2.0

        # ── Animation de révélation dramatique ────────────────────────────
        elite_js = "true" if is_elite else "false"
        _c = color
        _note = f"{note:.3f}"
        _pct = str(pct)
        _badge = f"{emoji}&nbsp;{label}"
        _plage = f"Plage PL : {NOTE_MIN:.2f} &rarr; {NOTE_MAX:.2f} &middot; RMSE 0.245"

        reveal_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Inter:wght@600&display=swap');
body{margin:0;padding:0;background:transparent;}
#rc{background:linear-gradient(135deg,#fff 0,#f5f3ff 100%);border:1.5px solid #e0e4f8;border-radius:18px;padding:1.5rem 1.2rem 1.1rem;text-align:center;box-shadow:0 6px 24px rgba(99,102,241,.12);position:relative;overflow:hidden;}
#sl{position:absolute;top:0;left:0;right:0;height:3px;animation:scan 1.8s ease-in-out forwards;opacity:0;}
@keyframes scan{0%{top:0%;opacity:1;}100%{top:100%;opacity:0;}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes pglow{0%,100%{box-shadow:0 6px 24px rgba(99,102,241,.12);}50%{box-shadow:0 0 50px rgba(99,102,241,.5);}}
.pulse{animation:pglow .8s ease-in-out 2;}
#lbl{color:#8b92c4;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;font-family:Inter,sans-serif;font-weight:600;margin:0 0 .3rem;}
#spn{margin:.2rem auto .4rem;width:28px;height:28px;border:3px solid #eef0ff;border-top:3px solid #6366f1;border-radius:50%;animation:spin .8s linear infinite;}
#sd{font-family:'Space Grotesk',sans-serif;font-size:4rem;font-weight:700;line-height:1;margin:0 0 .5rem;color:#ff5050;}
#bdg{display:inline-block;border-radius:30px;padding:4px 16px;font-size:.73rem;font-weight:700;letter-spacing:.1em;font-family:Inter,sans-serif;border:1.5px solid transparent;opacity:0;transform:scale(.6);transition:opacity .4s,transform .4s;}
#pt{background:#eef0ff;border-radius:8px;height:8px;margin:.9rem 0 .4rem;overflow:hidden;}
#pb{width:0%;height:100%;border-radius:8px;background:#ff5050;}
#pltxt{color:#b0b8d8;font-size:.6rem;margin:0;font-family:Inter,sans-serif;}
</style>
<div id="rc">
  <div id="sl"></div>
  <p id="lbl">Analyse en cours...</p>
  <div id="spn"></div>
  <div id="sd">0.000</div>
  <span id="bdg">""" + _badge + """</span>
  <div id="pt"><div id="pb"></div></div>
  <p id="pltxt">""" + _plage + """</p>
</div>
<script>
var T=""" + _note + """,TP=""" + _pct + """,FC='""" + _c + """',IE=""" + elite_js + """;
function gc(v){
  if(v<0.5)return'#ff5050';
  if(v<1.0)return'#ff8c42';
  if(v<1.5)return'#ffc400';
  if(v<2.0)return'#00c4ff';
  return'#00ff87';
}
function ease(t){return t===1?1:1-Math.pow(2,-10*t);}
var dur=2600,t0=null;
function step(ts){
  if(!t0){t0=ts;var s=document.getElementById('spn');if(s)s.style.display='none';}
  var p=Math.min((ts-t0)/dur,1),e=ease(p),cur=e*T,col=gc(cur);
  document.getElementById('sd').textContent=cur.toFixed(3);
  document.getElementById('sd').style.color=col;
  var pb=document.getElementById('pb');
  pb.style.width=(e*TP).toFixed(1)+'%';
  pb.style.background='linear-gradient(90deg,'+col+','+col+'88)';
  if(p<1){requestAnimationFrame(step);}
  else{
    document.getElementById('sd').textContent=T.toFixed(3);
    document.getElementById('sd').style.color=FC;
    pb.style.width=TP+'%';
    pb.style.background='linear-gradient(90deg,'+FC+','+FC+'88)';
    document.getElementById('lbl').textContent='Note efficacite predite';
    var b=document.getElementById('bdg');
    b.style.background=FC+'18';b.style.color=FC;
    b.style.borderColor=FC+'60';b.style.opacity='1';b.style.transform='scale(1)';
    document.getElementById('rc').classList.add('pulse');
    if(IE){
      var cols=['#00ff87','#6366f1','#ffc400','#00c4ff','#8b5cf6','#f59e0b'];
      for(var i=0;i<50;i++){(function(i){
        setTimeout(function(){
          var c=document.createElement('div');
          c.style.position='fixed';c.style.left=(Math.random()*100)+'%';c.style.top='0px';
          c.style.width=(6+Math.random()*6)+'px';c.style.height=(6+Math.random()*6)+'px';
          c.style.background=cols[i%cols.length];c.style.borderRadius='2px';
          c.style.pointerEvents='none';c.style.opacity='1';
          document.body.appendChild(c);
          var d2=(1800+Math.random()*1200),s2=performance.now();
          function fall(now){
            var p2=Math.min((now-s2)/d2,1);
            c.style.transform='translateY('+(p2*300)+'px) rotate('+(p2*720)+'deg)';
            c.style.opacity=''+(1-p2);
            if(p2<1)requestAnimationFrame(fall);else c.remove();
          }
          requestAnimationFrame(fall);
        },i*40);
      })(i);}
    }
  }
}
var _iv=setInterval(function(){
  if(document.getElementById('sd')){
    clearInterval(_iv);
    setTimeout(function(){requestAnimationFrame(step);},300);
  }
},30);
</script>
"""
        components.html(reveal_html, height=260)

    # ── Top 5 référence ────────────────────────────────────────────────────
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
        color:#8b5cf6;letter-spacing:0.14em;text-transform:uppercase;margin:1rem 0 0.3rem;'>
        🏆 &nbsp; Référence Premier League
    </p>
    """, unsafe_allow_html=True)
    st.caption("Top 5 joueurs du dataset")

    RANK_COLORS = ["#f59e0b", "#94a3b8", "#cd7c2f", "#6366f1", "#06b6d4"]
    for i, (name, score) in enumerate(REF_PLAYERS):
        bw   = int((score - NOTE_MIN) / (NOTE_MAX - NOTE_MIN) * 100)
        rcol = RANK_COLORS[i]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:0.55rem;"
            f"background:#ffffff;border-radius:10px;padding:0.45rem 0.8rem;"
            f"border:1px solid #eef0ff;box-shadow:0 2px 8px rgba(99,102,241,0.06);'>"
            f"<span style='font-size:0.7rem;font-weight:700;color:{rcol};"
            f"min-width:16px;font-family:Space Grotesk,sans-serif;'>#{i+1}</span>"
            f"<span style='min-width:80px;font-size:0.82rem;color:#374151;"
            f"font-family:Inter,sans-serif;font-weight:500;'>{name}</span>"
            f"<div style='flex:1;background:#eef0ff;border-radius:4px;height:6px;overflow:hidden;'>"
            f"<div style='width:{bw}%;height:100%;"
            f"background:linear-gradient(90deg,{rcol},{rcol}88);border-radius:4px;'></div></div>"
            f"<span style='min-width:34px;text-align:right;font-family:Space Grotesk,sans-serif;"
            f"font-size:0.92rem;font-weight:700;color:{rcol};'>{score:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Section "Votre joueur" avec bouton audio TTS ──────────────────────
    if predict_clicked:
        st.markdown("---")
        bw_me = int(max(0, (note - NOTE_MIN) / (NOTE_MAX - NOTE_MIN) * 100))

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"background:linear-gradient(135deg,#fffbeb,#fef3c7);"
            f"border-radius:12px;padding:0.6rem 0.9rem;"
            f"border:1.5px solid #fbbf24;box-shadow:0 3px 12px rgba(251,191,36,0.2);'>"
            f"<svg width='14' height='14' viewBox='0 0 20 20' fill='none'>"
            f"<polygon points='10,2 12,8 18,8 13,12 15,18 10,14 5,18 7,12 2,8 8,8' fill='#f59e0b'/>"
            f"</svg>"
            f"<span style='min-width:80px;font-size:0.82rem;color:#92400e;font-weight:700;"
            f"font-family:Inter,sans-serif;'>{nom_affiche}</span>"
            f"<div style='flex:1;background:#fde68a;border-radius:4px;height:6px;overflow:hidden;'>"
            f"<div style='width:{bw_me}%;height:100%;"
            f"background:linear-gradient(90deg,#f59e0b,#ef4444);border-radius:4px;'></div></div>"
            f"<span style='min-width:34px;text-align:right;font-family:Space Grotesk,sans-serif;"
            f"font-size:0.92rem;font-weight:700;color:#b45309;'>{note:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Bouton audio — components.v1.html() = vrai iframe, JS exécuté correctement
        commentaire_json = commentaire_tts.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')

        audio_html = f"""
        <style>
          body {{ margin:0; background:transparent; display:flex; justify-content:center; }}
          #tts-btn {{
            background: linear-gradient(135deg,#6366f1,#8b5cf6);
            color: #ffffff;
            font-family: 'Inter', Arial, sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.8rem;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35);
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 4px;
          }}
          #tts-btn:hover {{ transform: translateY(-2px); box-shadow: 0 8px 28px rgba(99,102,241,0.5); }}
        </style>
        <button id="tts-btn">🎙️ Écouter l'analyse</button>
        <script>
          var TEXTE = "{commentaire_json}";
          var btn   = document.getElementById('tts-btn');
          var synth = window.speechSynthesis;

          function loadVoices(cb) {{
            var v = synth.getVoices();
            if (v.length) {{ cb(v); return; }}
            synth.onvoiceschanged = function() {{ cb(synth.getVoices()); }};
          }}

          btn.addEventListener('click', function() {{
            if (synth.speaking) {{
              synth.cancel();
              btn.textContent = '🎙️ Écouter l\\'analyse';
              btn.style.background = 'linear-gradient(135deg,#6366f1,#8b5cf6)';
              return;
            }}
            loadVoices(function(voices) {{
              var utter  = new SpeechSynthesisUtterance(TEXTE);
              utter.lang = 'fr-FR';
              utter.rate = 0.88;
              utter.pitch = 1.0;
              utter.volume = 1.0;
              var frVoice = voices.find(function(v) {{ return v.lang.startsWith('fr'); }});
              if (frVoice) utter.voice = frVoice;
              utter.onstart = function() {{
                btn.textContent = '⏹️ Arrêter';
                btn.style.background = 'linear-gradient(135deg,#ef4444,#b91c1c)';
              }};
              utter.onend = function() {{
                btn.textContent = '🎙️ Écouter l\\'analyse';
                btn.style.background = 'linear-gradient(135deg,#6366f1,#8b5cf6)';
              }};
              synth.speak(utter);
            }});
          }});
        </script>
        """
        components.html(audio_html, height=60)

# ═══════════════════════════════════════════════════════════════════════════════
# RADAR CHART — affiché en pleine largeur sous les deux colonnes
# ═══════════════════════════════════════════════════════════════════════════════
if predict_clicked:
    st.markdown("---")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;
        color:#6366f1;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;'>
        🕸️ &nbsp; Profil radar — Analyse comparative
    </p>
    """, unsafe_allow_html=True)
    st.caption("Votre joueur vs E. Haaland vs Moyenne Premier League · valeurs normalisées [0-1]")

    # ── Normalisation des features sur leur plage max réelle ──────────────
    def normaliser(b, p, c, mb, mp):
        return [
            round(min(b  / 2.0,  1.0), 4),   # Buts/90      max ≈ 2.0
            round(min(p  / 2.0,  1.0), 4),   # Passes/90    max ≈ 2.0
            round(min(c  / 3.0,  1.0), 4),   # Contrib/90   max ≈ 3.0
            round(min((900 - mb) / 870, 1.0), 4),   # Eff. But   (moins = mieux → inverser)
            round(min((900 - mp) / 870, 1.0), 4),   # Eff. Passe (moins = mieux → inverser)
        ]

    CATEGORIES = ["Buts / 90", "Passes D. / 90", "Contrib. / 90", "Eff. But", "Eff. Passe"]

    # Haaland : stats réelles saison 23/24 (buts=1.05, pd=0.21, c=1.26, mb=86, mp=429)
    vals_haaland = normaliser(1.05, 0.21, 1.26, 86, 429)
    # Moyenne PL (valeurs par défaut de l'app)
    vals_moy     = normaliser(0.34, 0.20, 0.54, 268, 451)
    # Joueur analysé
    vals_joueur  = normaliser(buts, passes, contrib, min_but, min_passe)

    # Fermer le polygone en répétant le 1er point
    cats_closed   = CATEGORIES + [CATEGORIES[0]]
    j_closed      = vals_joueur  + [vals_joueur[0]]
    h_closed      = vals_haaland + [vals_haaland[0]]
    m_closed      = vals_moy     + [vals_moy[0]]

    fig = go.Figure()

    # ── Trace Haaland ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=h_closed, theta=cats_closed,
        fill="toself",
        name="E. Haaland",
        line=dict(color="#1D9E75", width=2, dash="dash"),
        fillcolor="rgba(29,158,117,0.10)",
        marker=dict(size=5, color="#1D9E75"),
    ))

    # ── Trace Moyenne PL ───────────────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=m_closed, theta=cats_closed,
        fill="toself",
        name="Moy. Premier League",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        fillcolor="rgba(148,163,184,0.08)",
        marker=dict(size=4, color="#94a3b8"),
    ))

    # ── Trace Joueur (au-dessus) ───────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=j_closed, theta=cats_closed,
        fill="toself",
        name=nom_affiche,
        line=dict(color="#6366f1", width=2.5),
        fillcolor="rgba(99,102,241,0.18)",
        marker=dict(size=6, color="#6366f1"),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(240,244,255,0.5)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor="rgba(99,102,241,0.15)",
                linecolor="rgba(99,102,241,0.2)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#374151", family="Inter"),
                gridcolor="rgba(99,102,241,0.12)",
                linecolor="rgba(99,102,241,0.2)",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#374151", family="Inter"),
            bgcolor="rgba(255,255,255,0.0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=30, b=60),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:2rem;">
    <div style="height:2px;background:linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4);
        border-radius:2px;margin-bottom:1rem;"></div>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
        <span style="font-family:Inter,sans-serif;font-size:0.72rem;color:#9ca3c8;letter-spacing:0.04em;">
            Projet Data Science · Master · Premier League 2023/2024 · ElasticNet Regression · Streamlit
        </span>
        <span style="font-family:Space Grotesk,sans-serif;font-size:0.75rem;font-weight:700;
            background:linear-gradient(135deg,#6366f1,#8b5cf6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            letter-spacing:0.1em;text-transform:uppercase;">
            ✦ &nbsp; ANOH AMON FRANCKLIN HEMERSON
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
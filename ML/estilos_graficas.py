
# Paleta de colores azules
COLORS = {
    'primary': '#1e3a8a',        # Azul oscuro principal
    'secondary': '#3b82f6',      # Azul medio
    'accent': '#60a5fa',         # Azul claro
    'background': '#dbeafe',     # Azul muy claro para fondos
    'gradient_start': '#1e40af', # Azul para gradientes
    'gradient_end': '#93c5fd',   # Azul claro para gradientes
    'mean_line': '#dc2626',      # Rojo para línea de media
    'grid': '#e0e7ff',           # Azul muy suave para grid
    'text': '#1e293b',           # Gris oscuro para texto
    'text_light': '#475569'      # Gris medio para texto secundario
}

# Configuración de estilos para matplotlib
PLOT_STYLE = {
    'figure': {
        'facecolor': '#f8fafc',
        'edgecolor': 'none',
        'tight_layout': {'pad': 3.0}
    },
    'axes': {
        'facecolor': 'white',
        'edgecolor': COLORS['primary'],
        'linewidth': 1.5,
        'labelcolor': COLORS['text'],
        'titleweight': 'bold',
        'titlesize': 11,
        'labelsize': 9,
        'grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.color': COLORS['grid']
    },
    'histogram': {
        'alpha': 0.8,
        'edgecolor': COLORS['primary'],
        'linewidth': 1.5,
        'color': COLORS['secondary']
    },
    'mean_line': {
        'color': COLORS['mean_line'],
        'linestyle': '--',
        'linewidth': 2.5,
        'alpha': 0.9
    },
    'text': {
        'color': COLORS['text'],
        'fontsize': 8,
        'fontweight': 'medium'
    },
    'legend': {
        'facecolor': 'white',
        'edgecolor': COLORS['grid'],
        'framealpha': 0.9,
        'fontsize': 8,
        'title_fontsize': 9,
        'borderpad': 0.5,
        'columnspacing': 1.0,
        'handlelength': 1.5
    },
    'title': {
        'fontsize': 18,
        'fontweight': 'bold',
        'color': COLORS['primary'],
        'pad': 20
    }
}

def apply_style(fig, ax):
    """
    Aplica el estilo personalizado a una figura y ejes
    """
    # Estilo de la figura
    fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])
    
    # Estilo de los ejes
    ax.set_facecolor(PLOT_STYLE['axes']['facecolor'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Grid
    ax.grid(True, alpha=PLOT_STYLE['axes']['grid.alpha'], 
            linestyle=PLOT_STYLE['axes']['grid.linestyle'],
            linewidth=PLOT_STYLE['axes']['grid.linewidth'],
            color=PLOT_STYLE['axes']['grid.color'])
    ax.set_axisbelow(True)
    
    # Tick parameters
    ax.tick_params(colors=COLORS['text_light'], labelsize=8)
    
def get_gradient_colors(n_bins):
    """
    Genera colores en gradiente para las barras del histograma
    """
    import matplotlib.cm as cm
    import numpy as np
    
    # Crear un colormap personalizado de azules
    cmap = cm.get_cmap('Blues')
    colors = [cmap(0.4 + 0.5 * i / (n_bins - 1)) for i in range(n_bins)]
    return colors

def style_histogram_bars(patches, values):
    """
    Aplica estilos a las barras del histograma con efecto gradiente
    """
    # values es un array numpy, usar .max() en lugar de max()
    max_val = values.max() if len(values) > 0 else 1
    colors = get_gradient_colors(len(patches))
    
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(COLORS['primary'])
        patch.set_linewidth(1.2)
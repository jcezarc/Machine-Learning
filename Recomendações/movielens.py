import pandas as pd
import numpy as np
from random import choices

def get_filmes():
    filmes = pd.read_csv('ml-latest-small/movies.csv')
    filmes.columns = ['filmeId', 'titulo', 'generos']
    return filmes

def get_notas():
    df_notas = pd.read_csv('ml-latest-small/ratings.csv')
    df_notas.columns = ['usuarioId', 'filmeId', 'nota', 'momento']
    # ======= Ajusta as notas para a escala de PONTUAÇÃO ===================
    df_notas['nota'] = df_notas['nota'] - 3
    # Assim, filmes ruins com muitos votos não ganham de filmes bons com menos votos
    #    <<--(1 péssimo)---(2 ruim)---(3 razoável)---(4 bom)---(5 ótimo)--->>
    #        -2 pontos      -1 ponto    0 pontos     1 ponto    2 pontos
    # ======================================================================
    return df_notas

def pontuacao_filmes(df_notas):
    result = df_notas.groupby('filmeId')['nota'].sum()
    return pd.DataFrame(result)

def assistidos_por(voce, df_notas=None):
    if df_notas is None:
        df_notas = get_notas()
    result = df_notas[
        (df_notas['usuarioId']==voce)
    ].set_index('filmeId')
    return list(result.index)

def genero_preferido(param, df_filmes):
    if isinstance(param, list):
        lista_filmes = param
    else:
        lista_filmes = assistidos_por(param)
    resumo = {}
    # --- Soma todas ocorrências de gêneros nos filmes: ---------
    generos = df_filmes[df_filmes['filmeId'].isin(lista_filmes)]['generos']
    for expr in generos:
        for nome_genero in expr.split('|'):
            ocorrencias = resumo.setdefault(nome_genero, 0) + 1
            resumo[nome_genero] = ocorrencias
    # ----- Encontra o nome do gênero com mais ocorrências ------
    encontrado = ''
    for nome_genero in resumo:
        ocorrencias = resumo[nome_genero]
        if not encontrado or ocorrencias > resumo[encontrado]:
            encontrado = nome_genero
    return encontrado

def notas_usuario(usuario_id, df_notas):
    return df_notas.query(f'usuarioId=={usuario_id}')[
        ['filmeId', 'nota']
    ].set_index('filmeId')

def distancia_usuarios(usuario_esq, usuario_dir, df_notas=None):
    if usuario_esq == usuario_dir:
        # ----> Não compara um usuário com ele mesmo!
        return None
        # -------------------------------------------    
    notas_esq = notas_usuario(usuario_esq, df_notas)
    notas_dir = notas_usuario(usuario_dir, df_notas)
    result = notas_esq.join(
        notas_dir,
        lsuffix='_esq',
        rsuffix='_dir'
    ).dropna()
    if len(result) < 3:
        # ----> Os usuários devem ter pelo menos 3 filmes em comum
        return None
        # -------------------------------------------------------------
    distancia = np.linalg.norm(result['nota_esq'] - result['nota_dir'])
    return {
        'usuario': usuario_dir,
        'distancia': distancia
    }

def mais_similares(voce, df_notas=None, tam_amostra=100, qtd_retornar=10):
    # --- Extrai uma amostra de todos os usuários ---
    outros_usuarios = choices(
        df_notas['usuarioId'].unique(),
        k=tam_amostra
    )
    result = []
    # -- Localiza os N mais próximos (onde N = qtd_retornar) ---
    for usuario_id in outros_usuarios:
        info = distancia_usuarios(voce, usuario_id, df_notas)
        if info is None:
            continue
        result.append(info)
    return pd.DataFrame(result).sort_values('distancia').head(qtd_retornar).set_index('usuario')

def recomendacoes(voce, df_notas=None, df_filmes=None, filtrar_genero=False):
    if df_notas is None:
        df_notas = get_notas()
    #----- Todos os filmes que vocÊ assistiu... ------
    assistidos = assistidos_por(voce, df_notas)
    #----- Lista os usuários similares a vocÊ: -----
    usuarios = list(mais_similares(voce, df_notas).index)
    #--- Filmes NÃO assistidos avaliados pelos usuários similares: ---
    result = df_notas[
        (~df_notas['filmeId'].isin(assistidos))
        &
        (df_notas['usuarioId'].isin(usuarios))
    ].drop(columns=['momento', 'usuarioId'])
    # --- Determina a pontuação dos filmes recomendados: ---
    result = pontuacao_filmes(result)
    #--- Junta com os nomes e outros dados dos filmes -----
    if df_filmes is None:
        df_filmes = get_filmes()
    if filtrar_genero:
        #=== Traz os filmes somente do gênero que vocÊ prefere ===
        df_filmes = df_filmes[df_filmes['generos'].str.contains(
            genero_preferido(assistidos, df_filmes)
        )]
        #=========================================================
    return result.join(
        df_filmes.set_index('filmeId')
    ).dropna().sort_values('nota', ascending=False).head(10)

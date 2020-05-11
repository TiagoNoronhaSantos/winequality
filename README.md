# winequality
Teste Lopes - Classificação de Vinhos
O código com o objetivo de fazer uma análise exploratória dos dados e aplicação de uma técnica de classificação foi desenvolvido em Python3.

Primeiramente, fez-se uma análise da consistência dos dados e possíveis espaços vazios dentro do banco disponível. Nenhum valor nulo foi encontrado, porém a variável "alcohol" apresentou inconsistências com valores que não pertencem àquela informação.

Optou-se então, por realizar uma exclusão das linhas do banco de dados onde tal variável apresentou inconsistência, pensou-se em utilizar a técnica de substituição pela média, mas não foi a forma escolhida.

Após isso, fez-se um filtro para eliminar outliers presentes no banco de dados, analisando-se todas as variáveis e excluindo os dados através da separação por quantis com margem de 10%, margem essa que pode ser alterada para futuras análises sem maiories dificuldades, é possível observar o filtro dos outliers olhando os gráficos boxplot gerados para cada variável do banco de dados tanto antes quanto após a aplicação do corte.

Pensando nos distintos níveis e características entre vinhos brancos e vermelhos, optou-se por trabalhar com eles separadamente, ciando-se dois conjuntos de dados.

Ao invés de aplicar de forma direta as técnicas de classificação, pensou-se em realizar uma busca por correlações para verificar quais variáveis estão possivelmente mais bem correlacionadas, para tanto, fez-se dois mapas de calor entre todas as variáveis do modelo, um para cada tipo de vinho.

Observando as correlações de todas as variáveis com os nossos "targets" (o nível de qualidade do vinho), atentou-se que, para o vinho vermelho, as variáveis mais correlatas à qualidade são: Sulphates e Alcohol. Para o vinho branco, encontrou-se: pH e Alcohol.

Decidiu-se então aplicar o método de classificação para apenas duas variáveis e posteriormente, para todo o conjunto de variáveis para avaliações posteriores.

Os modelos escolhidos para realizar a classificação dos vinhos baseados em suas características físico-químicas são: Regressão Linear e Rede Neural MLP (Perceptron de Múltiplas Camadas). A primeira menos robusta, porém eficaz para conjuntos de dados específicos, a segunda mais elaborada e potencialmente não-linear, na intenção de verificar como ambas técnicas tão distintas se comportam frente ao mesmo problema.

Outro processo utilizado, foi a normalização dos dados para possivelmente evitar alterações na resolução por possíveis grandezas siginificativas entre as variáveis do banco.

Sendo assim, fez-se a separação das rodadas em:
1) Regressão Linear Red Wine para Sulpheto e Alcohol
2) Regressão Linear Red Wine para todas as variáveis
3) Regressão Linear White Wine para pH e Alcohol
4) Regressão Linear White Wine para todas as variáveis
5) Regressão Linear Red Wine para Sulpheto e Alcohol com variáveis normalizadas
6) Regressão Linear Red Wine para todas as variáveis com variáveis normalizadas
7) Regressão Linear White Wine para pH e Alcohol com variáveis normalizadas
8) Regressão Linear White Wine para todas as variáveis com variáveis normalizadas
9) Rede neural MLP Red Wine para Sulpheto e Alcohol
10) Rede neural MLP Red Wine para todas as variáveis
11) Rede neural MLP White Wine para Sulpheto e Alcohol
12 Rede neural MLP White Wine para todas as variáveis

Para a obtenção dos resultados e escolher uma configuração satisfatória, utilizou-se o índice MSE (Erro Quadrático Médio) que apresenta a distância entre o conjunto de dados real e o conjunto estimado pelas técnicas. Tais índices estão a seguir na mesma ordem da numeração acima, justificando a escolha do melhor método e configuração.
1) MSE = 0.4483
2) MSE = 1.0328e-29
3) MSE = 0.6545
4) MSE = 5.3127e-29
5) MSE = 0.4483
6) MSE = 5.3127e-29
7) MSE = 0.6254
8) MSE = 7.2100e-31
9) MSE = 1.0572
10) MSE = 1.0572
11) MSE = 0.7898
12) MSE = 0.7898

Os melhores índices foram encontrados para Red Wine (MSE = 1.0328e-29), ou seja, todas as variáveis utilizadas no treinamento da técnica sem normalização dos dados. Para White Wine (MSE = 7.2100e-31), encontrou-se com a configuração de todas as variáveis normalizadas, e a técnica Regressão Linear.

A Rede Neural acabou não apresentando resultados tão bons quanto a Regressão Linear, motivo de mais pesquisa sobre os dados e configuração da Rede.

Observações: Não fez-se separação entre dados de treinamento e validação, todos os dados foram utilizados para treinamento. Ao rodar o código, somosa avisados por um "Warning" sobre a cópia de um DataFrame, o que não influencia na compilação, nem na execução ou resultados. O código gera algumas figuras e as salva.

OutputDerivada:  soma (InputDerivada * Peso)
QuantidadeErroDerivada: numero de derivadas acumuladas
ErroDerivadaAcumuladas: soma de todos ErroDerivada
ErroDerivada: inputDer * ultimoNode[Output]
QuantidadeInputDerivada: numero de InputDerivadaAcumuladas
InputDerivadaAcumuladas: soma de todos InputDerivada
InputDerivada: ativacao_derivada(somas_antes_da_ativacao) * OutputDerivada
UltimoLayer[OutputDerivada]: função_do_erro_derivada(output, previsto)
Output: ativação(somas_antes_da_ativacao)
somas_antes_da_ativacao: pesos * inputs + bias


Backpropagation: 

    Ultimo Layer | Cada neuronio ->
        OutputDerivada = função_do_erro_derivada(output, output_previsto)

    Ultimo layer para o segundo layer | neuronio por neuronio -> 
        InputDerivada = ativacao_derivada(somas_antes_da_ativacao) * OutputDerivada
        InputDerivadaAcumuladas += InputDerivada

        UltimoLayer[OutputDerivada] += InputDerivada * Peso

    Ultimo layer para o segundo layer | peso por peso -> 
        ErroDerivada = InputDerivada * NeuronioUltimoLayer[Output]
        ErroDerivadaAcumuladas += ErroDerivada

    Retorna => InputDerivadaAcumuladas, ErroDerivadaAcumuladas

Atualização Pesos:

    Ultimo Layer para o segundo layer | neuronio por neuronio ->
        Bias -= LearningRate * InputDerivadaAcumuladas

    Ultimo layer para o segundo layer | peso por peso -> 
        Peso -= LearningRate * ErroDerivadaAcumuladas

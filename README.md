# rede_neural
Backpropagation: 

    Ultimo Layer | Cada neuronio ->
        OutputDerivada = função_do_erro_derivada(output, output_previsto)

    Ultimo layer para o segundo layer | neuronio por neuronio -> 
        InputDerivada = ativacao_derivada(somas_antes_da_ativacao) * OutputDerivada
        InputDerivadaAcumuladas += InputDerivada 
        QuantidadeInputDerivada++

        UltimoLayer[OutputDerivada] += InputDerivada * Peso

    Ultimo layer para o segundo layer | peso por peso -> 
        ErroDerivada = InputDerivada * NeuronioUltimoLayer[Output]
        ErroDerivadaAcumuladas += ErroDerivada
        QuantidadeErroDerivada++


Atualização Pesos:

    Ultimo Layer para o segundo layer | neuronio por neuronio ->
        Bias -= LearningRate * InputDerivadaAcumuladas / QuantidadeInputDerivada

    Ultimo layer para o segundo layer | peso por peso -> 
        Peso -= (LearningRate / QuantidadeErroDerivada) * ErroDerivadaAcumuladas

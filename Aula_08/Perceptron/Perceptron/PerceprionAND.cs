using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace perceptron
{
    public class PerceptronAND
    {
        public readonly double[] w = new double[3];
        public readonly int[,] _matrizAprendizado = new int[4, 3];
        public double _net;

        //Sinais de Entrada
        int X1 = 0;
        int X2 = 1;
        //Resultado Esperado (Target)
        int TARGET = 2;
        //bias (Limiar de Ativação)
        public int BIAS = -1;
        public PerceptronAND()
        {
            //tabela AND
            _matrizAprendizado[0, X1] = 0;
            _matrizAprendizado[0, X2] = 0;
            _matrizAprendizado[0, TARGET] = 0;

            _matrizAprendizado[1, X1] = 0;
            _matrizAprendizado[1, X2] = 1;
            _matrizAprendizado[1, TARGET] = 0;

            _matrizAprendizado[2, X1] = 1;
            _matrizAprendizado[2, X2] = 0;
            _matrizAprendizado[2, TARGET] = 0;

            _matrizAprendizado[3, X1] = 1;
            _matrizAprendizado[3, X2] = 1;
            _matrizAprendizado[3, TARGET] = 1;

            w[0] = 0;
            w[1] = 0;
            w[2] = 0;
        }

        int Executar(int x1, int x2)
        {
            _net = (x1 * w[0]) + (x2 * w[1]) + ((BIAS) * w[2]);

            return (_net >= 0) ? 1 : 0;
        }

        public void Treinar()
        {
            var treinou = false;

            //CADA LOOP DO TREINAMENTO É CHAMADA DE UMA EPOCA
            while (!treinou)
            {
                treinou = true;
                
                for (var i = 0; i < _matrizAprendizado.GetLength(0); i++)
                {
                    var saida = Executar(_matrizAprendizado[i, X1], _matrizAprendizado[i, X2]);

                    if (saida != _matrizAprendizado[i, TARGET])
                    {
                        CorrigirPeso(i, saida);

                        treinou = false;
                    }
                }
            }
        }

        //Função STEP
        void CorrigirPeso(int i, int saida)
        {
            w[0] = w[0] + (1 * (_matrizAprendizado[i, TARGET] - saida) * _matrizAprendizado[i, X1]);
            w[1] = w[1] + (1 * (_matrizAprendizado[i, TARGET] - saida) * _matrizAprendizado[i, X2]);
            w[2] = w[2] + (1 * (_matrizAprendizado[i, TARGET] - saida) * (BIAS));
        }

        public int ClassificarAmostra(int[] amostra)
        {

            //aplicação da separação dos dados linearmente após aprendizado
            var u = amostra.Select((t, k) => w[k] * t).Sum();

            var y = LimiarAtivacao(u);


            return y;
        }

        private int LimiarAtivacao(double u)
        {
            return (u >= 0) ? 1 : 0;
        }
    }
}


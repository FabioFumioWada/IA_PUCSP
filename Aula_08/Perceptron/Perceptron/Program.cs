using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace perceptron
{
    public class Program
    {

        static PerceptronXOR p = new PerceptronXOR();
                
        public static void Main(string[] args)
        {
            //pesos antes do treinamento

            Console.WriteLine("PESOS ANTES DE TREINAR");
            p.w.ToList().ForEach(x => Console.WriteLine(x + ","));

            Console.WriteLine("\n");

            //efetua-se o treinamento da rede
            p.Treinar();

            Console.WriteLine("\n");

            //pesos ajustados após treinamento
            Console.WriteLine("PESOS APOS DE TREINAR");
            p.w.ToList().ForEach(x => Console.WriteLine(x + ","));

            //dados de entrada para rede treinada, 0 e 0 resulta em 0 (tabela and) -1 corresponde ao BIAS

            int[] amostra1 = { 0, 1, p.BIAS }; // 0 e 1 -> 0 Classe FALSE
            int[] amostra2 = { 1, 0, p.BIAS }; // 1 e 0 -> 0 Classe FALSE
            int[] amostra3 = { 0, 0, p.BIAS }; // 0 e 0 -> 0 Classe FALSE
            int[] amostra4 = { 1, 1, p.BIAS }; // 1 e 1 -> 1 Classe TRUE


            int y=p.ClassificarAmostra(amostra1);
            Console.WriteLine(y > 0 ? "Amostra da classe TRUE >= 0" : "Amostra da classe FALSE < 0");
            y = p.ClassificarAmostra(amostra2);
            Console.WriteLine(y > 0 ? "Amostra da classe TRUE >= 0" : "Amostra da classe FALSE < 0");
            y = p.ClassificarAmostra(amostra3);
            Console.WriteLine(y > 0 ? "Amostra da classe TRUE >= 0" : "Amostra da classe FALSE < 0");
            y = p.ClassificarAmostra(amostra4);
            Console.WriteLine(y > 0 ? "Amostra da classe TRUE >= 0" : "Amostra da classe FALSE < 0");

            Console.ReadKey();
        }

        

        
    }
}
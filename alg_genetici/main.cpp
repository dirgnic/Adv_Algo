#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <sstream>

using namespace std;

// structura pentru un individ
struct Individual {
    string bits;      // Reprezentarea pe biți a cromozomului
    int int_val;      // Valoarea numerică obținută (din șirul binar)
    double x;         // Valoarea reală din domeniul [a, b]
    double fitness;   // Valoarea funcției în punctul x
    double sel_prob;  // Probabilitatea de selecție
};

// funcția ce calculează valoarea polinomului: f(x) = A*x² + B*x + C
double evaluate(double x, const vector<double>& coeffs) {
    return coeffs[0]*x*x + coeffs[1]*x + coeffs[2];
}

// Calculează numărul de biți necesari, folosind precizia de d zecimale
int calc_chromosome_length(double a, double b, int d) {
    double num_points = (b - a) * pow(10, d) + 1;
    int L = (int)ceil(log2(num_points));
    return L;
}

// Decodifică un șir binar într-o valoare reală: x = a + (int_val/(2^L - 1))*(b - a)
pair<int, double> decode(const string &bits, double a, double b, int L) {
    int int_val = stoi(bits, nullptr, 2); //getting int val of binary string
    int max_int = (1 << L) - 1;
    double x = a + (static_cast<double>(int_val) * (b - a)) / max_int; //getting coresp val in our interval
    return make_pair(int_val, x);
}

// Inițializează populația: pentru fiecare individ se calculează
// reprezentarea pe biți, valoarea reală și fitness-ul
vector<Individual> init_population(int pop_size, double a, double b, int d, const vector<double>& coeffs, int &L, mt19937 &rng) {
    L = calc_chromosome_length(a, b, d);
    int max_int = (1 << L) - 1;
    uniform_int_distribution<int> dist(0, max_int);
    // unif_int_distr -> in standard stl, genereaza nr aleat in interv dat
    vector<Individual> population;

    for (int i = 0; i < pop_size; i++) {
        int val = dist(rng);
        string bits;
        for (int j = L - 1; j >= 0; j--) {
            bits.push_back(((val >> j) & 1) ? '1' : '0');
        }
        //transformam in sir de biti numarul aleatoriu

        double x = a + (static_cast<double>(val) * (b - a)) / max_int;

        //dam valorii noastre reale o "mapare"(corespondent) in intervalul [a, b]
        double fit = evaluate(x, coeffs); //calculam valoarea functiei in punctul x
        population.push_back({bits, val, x, fit, 0.0});
        /*
         Se creează un obiect de tipul Individual care conține:
         bits: reprezentarea pe biți a cromozomului.
         val: valoarea numerică obținută din codificarea pe biți.
         x: valoarea reală corespunzătoare în intervalul [a, b].
         fit: fitness-ul, calculat ca valoarea funcției pentru 𝑥
         0.0: valoarea inițială a probabilității de selecție (va fi determinată ulterior).
         */
    }
    return population;
}

// Calculează vectorul de probabilități cumulative pentru selecție
vector<double> calc_cumulative(vector<Individual> &population) {
    double total_fit = 0.0;
    for (auto &ind : population)
        total_fit += ind.fitness; //val fctiei in pctul x, adaug la total

    for (auto &ind : population)
        ind.sel_prob = (total_fit > 0.0) ? ind.fitness / total_fit : 0.0;
    // probabilitatea de selectie e raportul dintre fitness ul total si val
    // fctiei in pctul x

    vector<double> cumulative;
    double cm = 0.0;
    for (auto &ind : population) {
        cm += ind.sel_prob;
        cumulative.push_back(cm);
    }
    //constr vector cumulativ:
    //pentru fiecare individ, cm reprezintă suma probabilităților de
    //selecție pentru toți indivizii de la început până la individul curent

    return cumulative;
}

// Se efectuează selecția cu metoda ruletă utilizând căutarea binară
// Se păstrează și detaliile operației (pentru prima generație)
vector<Individual> selection(const vector<Individual> &population, const vector<double> &cumulative, int num_to_select, vector<string> &sel_details, mt19937 &rng) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<Individual> selected; // vector gol pt indivizii selectati
    for (int i = 0; i < num_to_select; i++) {
        double u = dist(rng); // generam nr aleatoare in interval

        // rolul vectorului cumulativ -> in cautarea binara
        // (avem sume partiale deci e crescator pe internal)

        auto it = lower_bound(cumulative.begin(), cumulative.end(), u);
        int idx = distance(cumulative.begin(), it);
        // it (iterator) -> cautam binar in vectorul cumulativ valoarea
        // (lower_bound caută cel mai mic element din vectorul cumulative care
        // nu este mai mic decât u)

        //deci;
        /*
         * cautam u -> result is in [c[idx]-1; c[idx])
         *
         */

        double lower_bound_val = (idx == 0 ? 0.0 : cumulative[idx - 1]);
        //
        ostringstream oss;
        oss << "u = " << fixed << setprecision(6) << u << " se încadrează în intervalul ["
            << lower_bound_val << ", " << cumulative[idx] << ") => se selectează cromozomul " << (idx + 1);
        sel_details.push_back(oss.str());
        selected.push_back(population[idx]);
    }
    return selected;
}

// Operația de încrucișare (crossover) – perechi de cromozomi (excluzînd elitistul)
// pot suferi încrucișare la un punct de tăietură aleator, cu probabilitatea dată
vector<Individual> crossover(vector<Individual> &population, double a, double b, int L, const vector<double> &coeffs, double p_crossover, vector<string> &cross_details, mt19937 &rng) {
    // Excludem individul elitist (ultimul din populație)
    vector<Individual> non_elite(population.begin(), population.end()-1);
    shuffle(non_elite.begin(), non_elite.end(), rng);

    //shuffle ->  asiguram că atunci când se formează perechi pentru crossover,
    // acestea sunt alese la întâmplare și nu după o ordine fixă.

    vector<Individual> new_pop;

    for (size_t i = 0; i < non_elite.size(); i += 2) {
        // Dacă numărul de indivizi din non_elite este impar, ultimul individ va fi tratat separat (după else).
        if (i + 1 < non_elite.size()) {
           // se formează perechi de părinți: parent1 și parent2.
            Individual parent1 = non_elite[i];
            Individual parent2 = non_elite[i+1];

            //generam o probabilit random intre 0 si 1, daca e mai mica decat p_crossover, facem crossover
            uniform_real_distribution<double> prob_dist(0.0, 1.0);
            double prob = prob_dist(rng);
            if (prob < p_crossover) {

                uniform_int_distribution<int> cp_dist(1, L - 1);

                // alegem un bit random (cp) intre 1 si nr de biti din codificare

                int cp = cp_dist(rng);
                ostringstream oss;
                oss << "Crossover între " << parent1.bits << " și " << parent2.bits << " la punctul " << cp;
                cross_details.push_back(oss.str());
                // Se formează copii prin schimbul de biți la punctul de rupere
                string child1_bits = parent1.bits.substr(0, cp) + parent2.bits.substr(cp);
                string child2_bits = parent2.bits.substr(0, cp) + parent1.bits.substr(cp);
                // => incrucisare clasica cu punct de rupere unic

                auto dec1 = decode(child1_bits, a, b, L);
                auto dec2 = decode(child2_bits, a, b, L);
                //fctia decode transforma sirul binar in valoarea numerica corespondenta
                // pair: (int_val, val in interv)

                Individual child1 = {child1_bits, dec1.first, dec1.second, evaluate(dec1.second, coeffs), 0.0};
                Individual child2 = {child2_bits, dec2.first, dec2.second, evaluate(dec2.second, coeffs), 0.0};
                ostringstream oss2;
                oss2 << "Rezultat crossover: " << child1.bits << " și " << child2.bits;
                cross_details.push_back(oss2.str());
                new_pop.push_back(child1);
                new_pop.push_back(child2);
            } else {
                ostringstream oss;
                oss << "Fără crossover pentru perechea " << parent1.bits << " și " << parent2.bits;
                cross_details.push_back(oss.str());
                new_pop.push_back(parent1);
                new_pop.push_back(parent2);
            }
        } else {
            new_pop.push_back(non_elite[i]);
        }
    }
    // Adăugăm individul elitist (ultimul din populația originală)
    new_pop.push_back(population.back());
    return new_pop;
}

// Operația de mutație: pentru fiecare bit al fiecărui cromozom (cu excepția elitistului),
// se inversează bitul cu probabilitatea dată

/*
 În mutație, pentru fiecare cromozom (cu excepția celui elitist),
 se parcurge fiecare bit și, pentru fiecare bit, se aplică o posibilitate de a-l inversa (de la „0” la „1” sau invers),
 în funcție de o probabilitate dată
 */
vector<Individual> mutation(vector<Individual> &population, double a, double b, int L, const vector<double> &coeffs, double p_mutation, vector<string> &mut_details, mt19937 &rng) {
    vector<Individual> new_pop;
    for (size_t idx = 0; idx < population.size(); idx++) {
        // Presupunem că ultimul individ este elitistul și nu muteaza
        if (idx == population.size() - 1) {
            new_pop.push_back(population[idx]);
            continue;
        }
        string original = population[idx].bits;
        string mutated = original;
        for (int i = 0; i < L; i++) { //trecem prin toti bitii cromozomului, le dam switch cu o anume probabilitate
            //constr cromozom nou
            uniform_real_distribution<double> prob_dist(0.0, 1.0);
            if (prob_dist(rng) < p_mutation) {
                char old_bit = mutated[i];
                mutated[i] = (mutated[i] == '0') ? '1' : '0';
                ostringstream oss;
                oss << "Mutație: " << original << " – bitul de la poziția " << i
                    << " schimbat din " << old_bit << " în " << mutated[i];
                mut_details.push_back(oss.str());
            }
        }
        auto dec = decode(mutated, a, b, L); //calculam val cromozom (x intreg si in interval) dupa mutatie, adaugam iar in populatie
        Individual new_ind = {mutated, dec.first, dec.second, evaluate(dec.second, coeffs), 0.0};
        new_pop.push_back(new_ind);
    }
    return new_pop;
}

// funcția principală a algoritmului genetic
void genetic_algorithm(int pop_size, double a, double b, const vector<double> &coeffs, int precision_digits, double p_crossover_percent, double p_mutation_percent, int num_generations) {
    // conversia probabilităților din procente in fracții
    double p_crossover = p_crossover_percent / 100.0;
    double p_mutation = p_mutation_percent / 100.0;

    random_device rd;
    mt19937 rng(rd());

    int L = 0; // lungimea cromozomului
    vector<Individual> population = init_population(pop_size, a, b, precision_digits, coeffs, L, rng);
    //aflam populatia initiala
    vector<string> output_lines;

    output_lines.push_back("Populația Inițială: ");
    for (size_t i = 0; i < population.size(); i++) {
        ostringstream oss;
        oss << "Individ " << (i+1) << ": B = " << population[i].bits
            << ", X = " << fixed << setprecision(6) << population[i].x
            << ", f(X) = " << population[i].fitness;
        output_lines.push_back(oss.str());
    }

    // Calculăm probabilitățile de selecție și vectorul cumulativ
    vector<double> cumulative = calc_cumulative(population);

    output_lines.push_back("\nProbabilități de Selecție: ");
    for (size_t i = 0; i < population.size(); i++) {
        ostringstream oss;
        oss << "Individ " << (i+1) << ": f(X) = " << fixed << setprecision(6) << population[i].fitness
            << ", p = " << population[i].sel_prob; // scriem in fisier probabilitatea de selectie pt fiecare cromozom
        output_lines.push_back(oss.str());
    }


   //pentru fiecare individ, probabilit cumlativa reprezintă suma probabilităților de selecție pentru
   //toți indivizii de la început până la individul curent.
    output_lines.push_back("\nProbabilități Cumulate: ");
    output_lines.push_back("q0 = 0");
    for (size_t i = 0; i < cumulative.size(); i++) {
        ostringstream oss;
        oss << "q" << (i+1) << " = " << fixed << setprecision(6) << cumulative[i];
        output_lines.push_back(oss.str());
    }

    // Procesul de selecție (prima generație): se selectează pop_size - 1 indivizi
    output_lines.push_back("\nProcesul de Selecție (prima generație): ");
    vector<string> sel_details;
    vector<Individual> selected = selection(population, cumulative, pop_size - 1, sel_details, rng);
    //selectam
    for (auto &s : sel_details)
        output_lines.push_back(s);

    // Determinăm individul elitist (cu fitness maxim) -> pastrat intact in generatiile urm
    auto elitism_it = max_element(population.begin(), population.end(), [](const Individual &a, const Individual &b) {
        return a.fitness < b.fitness;
    });
    Individual elite = *elitism_it;
    ostringstream oss_elite;
    oss_elite << "\nElitism: Individul cu cel mai mare fitness este: B = " << elite.bits
              << ", X = " << fixed << setprecision(6) << elite.x
              << ", f(X) = " << elite.fitness;
    output_lines.push_back(oss_elite.str());
    selected.push_back(elite);  // adăugăm elitistul
    vector<Individual> new_population = selected;

    // Încrucișarea pentru prima generație (se afișează detalii)
    output_lines.push_back("\nProcesul de Încrucișare (prima generație): ");
    vector<string> cross_details;
    new_population = crossover(new_population, a, b, L, coeffs, p_crossover, cross_details, rng);
    for (auto &s : cross_details)
        output_lines.push_back(s);

    output_lines.push_back("\nPopulația după Încrucișare:");
    for (size_t i = 0; i < new_population.size(); i++) {
        ostringstream oss;
        oss << "Individ " << (i+1) << ": B = " << new_population[i].bits
            << ", X = " << fixed << setprecision(6) << new_population[i].x
            << ", f(X) = " << new_population[i].fitness;
        output_lines.push_back(oss.str());
    }

    // Mutația pentru prima generație (se afișează detalii)
    output_lines.push_back("\n Procesul de Mutație (prima generație): ");
    vector<string> mut_details;
    new_population = mutation(new_population, a, b, L, coeffs, p_mutation, mut_details, rng);
    for (auto &s : mut_details)
        output_lines.push_back(s);

    output_lines.push_back("\nPopulația după Mutație:");
    for (size_t i = 0; i < new_population.size(); i++) {
        ostringstream oss;
        oss << "Individ " << (i+1) << ": B = " << new_population[i].bits
            << ", X = " << fixed << setprecision(6) << new_population[i].x
            << ", f(X) = " << new_population[i].fitness;
        output_lines.push_back(oss.str());
    }

    // Rezumatul pentru generația 0
    int gen = 0;
    double max_fit = 0.0, sum_fit = 0.0;
    for (auto &ind : new_population) {
        sum_fit += ind.fitness;
        if (ind.fitness > max_fit)
            max_fit = ind.fitness;
    }
    double mean_fit = sum_fit / new_population.size(); //calculam media fitness ului si fitness ul maxim pt gen 0
    ostringstream oss_gen;
    oss_gen << "\n Rezumat Generația " << gen << ": "
            << "\nMax Fitness = " << fixed << setprecision(6) << max_fit
            << ", Mean Fitness = " << mean_fit;
    output_lines.push_back(oss_gen.str());

    // Setăm populația curentă pentru iterațiile următoare
    population = new_population;

    // Pentru generațiile ulterioare se afișează doar rezumatul (maxim și media fitness)
    for (gen = 1; gen < num_generations; gen++) {
        cumulative = calc_cumulative(population);
        vector<string> dummy_sel;
        vector<Individual> selected_gen = selection(population, cumulative, pop_size - 1, dummy_sel, rng);
        // Adăugăm elitistul
        auto best_it = max_element(population.begin(), population.end(), [](const Individual &a, const Individual &b) {
            return a.fitness < b.fitness;
        });
        Individual best = *best_it;
        selected_gen.push_back(best);
        vector<string> dummy_cross;
        new_population = crossover(selected_gen, a, b, L, coeffs, p_crossover, dummy_cross, rng);
        vector<string> dummy_mut;
        new_population = mutation(new_population, a, b, L, coeffs, p_mutation, dummy_mut, rng);
        population = new_population;
        max_fit = 0.0;
        sum_fit = 0.0;
        for (auto &ind : population) {
            sum_fit += ind.fitness;
            if (ind.fitness > max_fit)
                max_fit = ind.fitness;
        }
        mean_fit = sum_fit / population.size();
        ostringstream oss_summary;
        oss_summary << "\n Rezumat Generația " << gen << ": "
                    << "\nMax Fitness = " << fixed << setprecision(6) << max_fit
                    << ", Mean Fitness = " << mean_fit;
        output_lines.push_back(oss_summary.str());
    }

    // Scriem toate informațiile în fișierul Evolutie.txt
    ofstream fout("Evolutie.txt");
    for (auto &line : output_lines)
        fout << line << "\n";
    fout.close();

    cout << "Algoritmul genetic s-a terminat. Vezi fișierul 'Evolutie.txt' pentru detalii." << endl;
}

int main() {
    // Citim parametrii de la tastatură:
    // 1. Dimensiunea populației
    // 2. Capetele intervalului [a, b]
    // 3. Cei trei coeficienți ai polinomului
    // 4. Precizia (numărul de zecimale)
    // 5. Probabilitatea de crossover (în procente)
    // 6. Probabilitatea de mutație (în procente)
    // 7. Numărul de etape (generații)

    /*
    int population_size;
    double a, b;
    vector<double> coeffs(3);
    int precision_digits;
    double p_crossover_percent, p_mutation_percent;
    int num_generations;

    cin >> population_size;
    cin >> a >> b;
    cin >> coeffs[0] >> coeffs[1] >> coeffs[2];
    cin >> precision_digits;
    cin >> p_crossover_percent;
    cin >> p_mutation_percent;
    cin >> num_generations;
*/
    // parametri de intrare - exemplu
    int dim_pop = 20; // nr comozomi
    double a = -1.0, b = 2.0; // intervalul [a, b]
    vector<double> coeffs = {-1.0, 1.0, 2.0}; // pentru funcția -x^2 + x + 2
    double precision_digits = 6; // precizie (6 zecimale)
    double p_crossover = 25; // probabilitatea de crossover (25%)
    double p_mutation = 1;  // probabilitatea de mutație (1%)
    int num_generations = 50;

    genetic_algorithm(dim_pop, a, b, coeffs, precision_digits, p_crossover, p_mutation, num_generations);

    //fisier output:
    /*
    𝐵𝑖 - reprezentarea pe biți a cromozomului;
    𝑋𝑖 - val coresp cromozomului în domeniul de definiție al funcției
    𝑓(𝑋𝑖) - valoarea cromozomului, adică valoarea funcției în punctul din domeniu
    care corespunde acestuia
    */
    /*
     probabilit selectie pt fiecare cromozom
     probabilit cumulate care dau intervalele pentru selecție
     */
    /*
    procesul de selectie:  generarea unui
    număr aleator 𝑢 uniform pe [0, 1) și determinarea intervalului
    [𝑞𝑖, 𝑞𝑖+1) căruia aparține acest număr; corespunzător acestui in-
    terval se va selecta cromozomul 𝑖 + 1. -> repetam pana avem nr dorit de cromozomi.
    căutarea intervalului corespunzător se va face folosind
    căutarea binară.


    Evidențierea cromozomilor care participă la recombinare.
•   Pentru recombinările care au loc se evidențiază perechile de cromozomi care
    participă la recombinare, punctul de rupere generat
    aleator precum și cromozomii rezultați în urma recombinării (sau,
    după caz, se evidențiază tipul de încrucișare ales).

    Populația rezultată după mutațiile aleatoare.
•   Pentru restul generațiilor (populațiile din etapele următoare) se
    va afișa doar valoarea maximă și valoarea mediei a fitness-ului (performanței)
    populației.
    */

    return 0;
}

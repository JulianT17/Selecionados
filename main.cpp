#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include "ManoplaLelyBBB.h"
#include <armadillo>
#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

// Definindo o tipo 'Clock' para simplificar o uso de relógio de alta resolução
using Clock = std::chrono::high_resolution_clock;

// Função que calcula o valor de 'event' baseado em parâmetros específicos, usada para ajustar o comportamento do sistema
double event(double k, double a = 16, double b = 0.01) {
    return a * std::pow((1 - b), k);
}

// Função 'super_event' que faz cálculos para eventos, incluindo normalização e outras constantes
double super_event(double x_norm, double matrix_norm, double k, double a = 30, double b = 0.001, double sigma = 0.01) {
   return sigma * x_norm / matrix_norm + event(k, a, b);
}

// Definindo constantes para os argumentos da linha de comando (índices de posição para simplificar o acesso)
constexpr unsigned int ARG_CAN_INTERFACE = 1;
constexpr unsigned int ARG_VERSION = 1;
constexpr unsigned int ARG_SYNCCALLBACK = 2; // indica o tipo de callback (ex.: modo de posição, PID, etc.)

// Constantes específicas para valores de ganho (PID) e tipos de callback (usado em DLQR)
constexpr unsigned int ARG_SYNCCALLBACK_PID_KP = 3; // ganho proporcional do PID
constexpr unsigned int ARG_SYNCCALLBACK_PID_KD = 4; // ganho derivativo do PID
constexpr unsigned int ARG_SYNCCALLBACK_PID_KI = 5; // ganho integral do PID

// Valores padrão para os ganhos PID
constexpr float DEFAULT_KP = 1800.f;
constexpr float DEFAULT_KD = 150.f;
constexpr float DEFAULT_KI = 0.f;

// Enumeração que define os tipos de referência para o controle (zero constante ou onda senoidal)
enum class ReferenceType {
    ConstantZero,
    SineWave,
};

// Definição de macros para simplificar a leitura dos tipos de referência
#define CONSTANTZERO "constant-zero"
#define SINEWAVE "sine-wave"

// Sobrecarga do operador << para imprimir os tipos de referência de forma legível
std::ostream& operator<<(std::ostream& o, ReferenceType rt) {
    if (rt == ReferenceType::ConstantZero) {
        o << CONSTANTZERO;
    } else if (rt == ReferenceType::SineWave) {
        o << SINEWAVE;
    }
    return o;
}

// Mapeamento de strings para os tipos de referência
std::map<std::string, ReferenceType> ReferenceTypeString{
    {CONSTANTZERO, ReferenceType::ConstantZero},
    {SINEWAVE, ReferenceType::SineWave},
};

// Valor padrão para o tipo de referência
const ReferenceType DEFAULT_REFERENCETYPE = ReferenceType::ConstantZero;

// Declaração de matrizes e vetores do sistema, usados no cálculo do controlador DLQR
arma::mat sys_matrix_A = {{9.98609061e-01, 1.09025360e+01},
                          {-1.13123668e-04, 9.77691957e-01}};
arma::colvec sys_matrix_B = {8.6104e-3, 8.0799e-4};
arma::rowvec sys_dlqr_K = {-2.70079861, -233.77001218};

// Estrutura para armazenar as configurações de controle DLQR
struct EventDLQRControlPreset {
    arma::rowvec K;
    double a;
    double b;
    double sigma;
    ReferenceType ct;
};

// Enumeração dos diferentes perfis de controlador predefinidos (DR6, EDR24, etc.)
enum class ControllerPreset {
    DR6,
    EDR24,
    EDR83,
    DC6,
    EDC56,
    EDC103,
};

// Sobrecarga do operador << para imprimir o nome do perfil de controlador
std::ostream& operator<<(std::ostream& o, ControllerPreset cp) {
    switch (cp) {
        case ControllerPreset::DR6: o << "DR6"; break;
        case ControllerPreset::EDR24: o << "EDR24"; break;
        case ControllerPreset::EDR83: o << "EDR83"; break;
        case ControllerPreset::DC6: o << "DC6"; break;
        case ControllerPreset::EDC56: o << "EDC56"; break;
        case ControllerPreset::EDC103: o << "EDC103"; break;
    }
    return o;
}

// Mapeamento de strings para os perfis de controlador
std::map<std::string, ControllerPreset> ControllerPresetString{
    {"DR6", ControllerPreset::DR6},
    {"EDR24", ControllerPreset::EDR24},
    {"EDR83", ControllerPreset::EDR83},
    {"DC6", ControllerPreset::DC6},
    {"EDC56", ControllerPreset::EDC56},
    {"EDC103", ControllerPreset::EDC103},
};

// Estrutura para mapear cada perfil de controlador para uma configuração de controle específica
std::map<ControllerPreset, EventDLQRControlPreset> DLQRControlPresets{
    {ControllerPreset::DR6, EventDLQRControlPreset{{-89.9178, -1155.41}, 0, 0, 0, ReferenceType::ConstantZero}},
    {ControllerPreset::EDR24, EventDLQRControlPreset{{-1.90978, -194.491}, 16, 1e-3, 0.01, ReferenceType::ConstantZero}},
    // Outros perfis seguem o mesmo padrão...
};

bool controllerPresetSelected{false};
double matrix_norm{};

// Estrutura para armazenar pontos de dados de um log
using DataPoint = std::tuple<uint64_t, int32_t, double, int16_t, int32_t, float, float, float, float>;

// Classe 'MyLog' para registrar e salvar dados de controle em um arquivo
class MyLog {
    std::vector<DataPoint> log;
    std::vector<std::string> logHeader;
public:
    explicit MyLog(const std::vector<std::string>& logHeader) : log(), logHeader(logHeader) {}

    // Adiciona um ponto de dados ao log
    void addDataPoint(const DataPoint &dp) {
        log.push_back(dp);
    }

    // Salva os dados registrados em um arquivo
    void saveToFile(const std::string& fileName,
                    const std::string& controllerType,
                    const ReferenceType& rt,
                    arma::mat A,
                    arma::mat B,
                    arma::mat K,
                    double a,
                    double b,
                    double sigma,
                    float refresh_rate) {
        using std::setw;
        using std::setfill;
        std::stringstream ssFileName;
        auto now = Clock::to_time_t(Clock::now());
        auto localtime = std::localtime(&now);

        // Nome do arquivo é gerado com data e hora atuais
        ssFileName << "log_" << localtime->tm_year + 1900 << "-" << setfill('0') << setw(2) << localtime->tm_mon+1
                   << "-" << setfill('0') << setw(2) << localtime->tm_mday << "_" << setfill('0') << setw(2) << localtime->tm_hour
                   << "-" << setfill('0') << setw(2) << localtime->tm_min << "-" << setfill('0') << setw(2) << localtime->tm_sec
                   << "_" << fileName;

        std::ofstream file{ssFileName.str()};

        // Cabeçalho do arquivo com configurações
        file.precision(10);
        file << "[Configuration]\n" << "controller_type = " << controllerType << "\n"
             << "reference_type = " << rt << "\n" << "A = [[ " << std::scientific << A[0,0] << " , " << A[0,1] << " ] , [ " << A[1,0] << " , " << A[1,1] << " ] ] \n"
             << "B = [[" << B[0] << "] , [" << B[1] << "]]\n"
             << "K = [[" << K[0] << " , " << K[1] << "]]\n"
             << "event_a = " << a << "\n" << "event_b = " << b << "\n" << "event_sigma = " << sigma << "\n"
             << "refresh_rate = " << refresh_rate << "\n" << "\n[Table]\n";

        // Salva cada ponto de dados registrado no log
        for(const auto& dp : log) {
            file << setw(20) << std::get<0>(dp) << ',' << setw(20) << std::get<1>(dp) << ',' << setw(20) << std::get<2>(dp) << ','
                 << setw(20) << std::get<3>(dp) << ',' << setw(20) << std::get<4>(dp) << ',' << setw(20) << std::get<5>(dp) << ','
                 << setw(20) << std::get<6>(dp) << ',' << setw(20) << std::get<7>(dp) << ',' << setw(20) << std::get<8>(dp) << '\n';
        }
    }
};


int main(int argc, char* argv[]) {
    std::cout << "Hello, World!" << std::endl;

    // Declaração de variáveis que serão usadas para armazenar argumentos e configurações.
    std::string can_interface_name{};
    std::string synccallback{};
    float kp{0}, ki{0}, kd{0}; // Ganhos de controle PID
    ReferenceType referenceType{}; // Tipo de referência para o controlador
    std::string controllerArgParameter;

    // Parâmetros para o cálculo de eventos
    double event_a{30};
    double event_b{0.001};
    double event_sigma{0.01};

    std::string version{};
    std::stringstream usage_message;

    // Mensagem de uso, exibida caso o usuário forneça argumentos incorretos ou peça ajuda.
    usage_message << "Usage:\n"
                  << argv[0] << "<canbus-interface-name> <canopen-syncsallback> [reference-type]\n"
                  << "  canopen-synccallback: position-mode, current-mode, pid, dlqr, dlqr-event\n"
                  << "    position-mode: no arguments\n"
                  << "    current-mode: no arguments\n"
                  << "    pid:\n"
                  << "      no arguments (default: kp=1800, kd=150, ki=0), or\n"
                  << "      kp kd ki\n"
                  << "    dlqr [controller-type]\n"
                  << "      reference-types:\n"
                  << "        no arguments (default: constant-zero), or\n"
                  << "        constant-zero\n"
                  << "        senoidal\n"
                  << "        DR6\n"
                  << "        DC6\n"
                  << "    dlqr-event [controller-type]\n"
                  << "      reference-types:\n"
                  << "        no arguments (default: constant-zero), or\n"
                  << "        constant-zero\n"
                  << "        senoidal\n"
                  << "        EDR24\n"
                  << "        EDR83\n"
                  << "        EDC56\n"
                  << "        EDC103\n"
                  << "\n"
                  << "ADVANCED: DON'T MESS WITH THIS IF YOU DON'T KNOW WHAT YOU ARE DOING!\n"
                  << "environment variable:"
                  << "    SYS_DLQR_K: defines the gain matrix for the dLQR controller\n"
                  << "      default (and stable):\n"
                  << "      SYS_DLQR_K=-2.70079861 -233.77001218\n"
                  << "    EVENT_A_B_SIGMA: defines the a, b and sigma parameters for triggering the event\n"
                  << "                     setting this will override the controller parameters!\n"
                  << "      (menor metrica_comparacao - default\n"
                  << "         EVNET_A_B_SIGMA=10 0.2 0.01\n"
                  << "      (menor update_rate)\n"
                  << "         EVNET_A_B_SIGMA=16 0.001 0.01\n"
                  << "\n";

    // Verificação inicial: Certifica-se de que o número correto de argumentos foi fornecido.
    if (argc >= 3) {
        can_interface_name = argv[ARG_CAN_INTERFACE];
        std::cout << "CAN interface escolhida: " << can_interface_name << std::endl;

        synccallback = argv[ARG_SYNCCALLBACK];
        std::cout << "SyncCallback escolhido: " << synccallback << std::endl;

        // Caso o sync seja "pid", processa os ganhos PID (P, D, I).
        if(synccallback == "pid") {
            if (argc < 5 && argc > 3) {
                std::cout << "You must give:\n"
                          << "\t- OR exactly 3 numbers (gains P, D and I) as argument to callback pid choosed!\n"
                          << "\t- OR no arguments to use default values for the gains (P:1800, D:150, I:0)\n";
                std::exit(1);
            }

    // Configura os parâmetros do controlador DLQR usando um perfil de controlador, se selecionado
    if (controllerPresetSelected) {
        event_a = DLQRControlPresets[ControllerPresetString[controllerArgParameter]].a;
        event_b = DLQRControlPresets[ControllerPresetString[controllerArgParameter]].b;
        sys_dlqr_K = DLQRControlPresets[ControllerPresetString[controllerArgParameter]].K;
        event_sigma = DLQRControlPresets[ControllerPresetString[controllerArgParameter]].sigma;
        referenceType = DLQRControlPresets[ControllerPresetString[controllerArgParameter]].ct;
    }

    std::stringstream ss;

    // Lê variáveis de ambiente para sobrescrever os valores de 'a', 'b' e 'sigma' (se existirem)
    if(const char* env_a_b_sigma = std::getenv("EVENT_A_B_SIGMA")) {
        std::cout << "OVERRIDING A, B and SIGMA with values from environment variable EVENT_A_B_SIGMA\n";
        ss = std::stringstream{env_a_b_sigma};
        ss >> event_a >> event_b >> event_sigma;
        std::cout << "NEW values:\n"
                  << "a       = " << event_a << "\n"
                  << "b       = " << event_b << "\n"
                  << "sigma   = " << event_sigma << "\n\n";
    } else {
        if (controllerPresetSelected) {
            std::cout << "Setting values from " << controllerArgParameter << " preset for a, b and sigma!\n";
        }
    }

    // Limpa o conteúdo de 'ss' para reutilizá-lo
    ss.str("");

    // Lê a variável de ambiente 'SYS_DLQR_K' para definir ganhos da matriz 'K' do controlador DLQR
    if(const char* env_k = std::getenv("SYS_DLQR_K")) {
        std::cout << "OVERRIDING K controller gains with values from environment variable SYS_DLQR_K\n";
        ss = std::stringstream{env_k};
        ss >> sys_dlqr_K[0] >> sys_dlqr_K[1];
        std::cout << "NEW values:\n"
                  << "K[0,0]  = " << sys_dlqr_K[0] << "\n"
                  << "K[0,1]  = " << sys_dlqr_K[1] << "\n\n";
    } else {
        if (controllerPresetSelected) {
            std::cout << "Setting default values for K!\n";
        }
    }

    // Imprime as matrizes de controle e a norma do produto 'B * K'
    sys_matrix_A.print("matrix A");
    sys_matrix_B.print("matrix B");
    sys_dlqr_K.print("dLQR controller matrix K");
    matrix_norm = arma::norm(sys_matrix_B * sys_dlqr_K, 2);
    std::cout << "norm of B*K: " << matrix_norm << "\n\n";

    // Imprime os parâmetros do controle e do evento
    std::cout << "control type: " << referenceType << "\n\n";
    std::cout << "Event parameters:\n"
              << "      a = " << event_a << "\n"
              << "      b = " << event_b << "\n"
              << "  sigma = " << event_sigma << "\n\n";

    // Instancia o objeto 'manopla' com a interface CAN selecionada
    manopla::ManoplaLelyBBB manopla{can_interface_name};
    auto initial_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count();

    // Configuração de log para registrar dados do sistema de controle
    MyLog log{{"time_us", "pulse_qc", "setpoint_current_mA", "actual_current_mA", "epos_velocity_unfiltered_rpm", "calculated_velocity_rad/s", "tracked_reference", "event_max_error", "event_error"}};

    // Função de callback para o modo de posição
    manopla::SyncCallback onSyncCallbackPositionMode = [&](const manopla::Time& t, const manopla::MotorInfo& mi, manopla::MyDriver& driver){
        driver.tpdo_mapped[0x6040][0] = static_cast<uint16_t>(0x02); // Mantém a tensão ativa
        if (t.current_dt_us > std::chrono::duration_cast<std::chrono::microseconds>(500ms).count()) {
            std::cout << setw(20) << t.sum_total_dt << setw(20) << t.current_dt_us << setw(20) << mi.currentPulses << '\n';
        }

        // Configura uma posição como senoide
        auto omega = (8 * 2 * M_PIf32);
        auto time = t.current_us - initial_time;
        auto sine_position = 500 * std::sin(static_cast<double>(time) * omega);

        driver.tpdo_mapped[0x2062][0] = static_cast<int32_t>(sine_position);
        if (time > std::chrono::duration_cast<std::chrono::microseconds>(100ms).count()) {
            std::cout << "sine: " << setw(20) << time << setw(20) << sine_position << "\n";
        }
    };

    uint64_t good_print_period_ms = 300;
    uint64_t dt_sum_for_printing_ms = 0;

    // Função de callback para o modo de corrente
    manopla::SyncCallback onSyncCallbackCurrentMode = [&](const manopla::Time& t, const manopla::MotorInfo& mi, manopla::MyDriver& driver) {
        driver.tpdo_mapped[0x6040][0] = static_cast<uint16_t>(0x02); // Mantém a tensão ativa
        auto omega = (2 * M_PIf32);
        auto time_s = static_cast<double>(t.current_us - initial_time) / 1000000.f;
        auto sine_current_setpoint = 400 * std::sin(time_s * omega);

        driver.tpdo_mapped[0x2030][0] = static_cast<int16_t>(sine_current_setpoint);

        // Calcula a velocidade e imprime dados periodicamente
        float calculated_speed = (mi.currentAngle - mi.prevAngle) / (static_cast<float>(t.current_dt_us) / 1000000.f);
        dt_sum_for_printing_ms += t.current_dt_us;
        if (dt_sum_for_printing_ms > good_print_period_ms) {
            std::cout << "t.current_us: " << setw(10) << t.current_us
                      << " time_us: " << setw(10) << time_s
                      << " calculated_speed: " << setw(10) << calculated_speed
                      << " currentAngle:" << setw(10) << mi.currentAngle
                      << " prevAngle:" << setw(10) << mi.prevAngle << '\n';
            dt_sum_for_printing_ms = 0ull;
        }

        // Registra os dados no log se o driver estiver pronto
        if(driver.IsReady()) {
            log.addDataPoint(DataPoint{time_s * 1000000, mi.currentPulses, sine_current_setpoint, mi.currentCurrent, mi.currentRotationUnfiltered, calculated_speed, 0, 0, 0});
        }
    };

    float referencePosition = 0.f;
    float referenceVelocity = 0.f;
    float errorPosition = 0.f;
    float errorPositionSum = 0.f;
    float errorVelocity = 0.f;
    float controlActionP = 0.f;
    float controlActionI = 0.f;
    float controlActionD = 0.f;
    float controlSignal = 0.f;
    float calculatedSpeed = 0.f;

    // Função de callback para o modo PID
    manopla::SyncCallback onSyncCallbackPID = [&](const manopla::Time& t, const manopla::MotorInfo& mi, manopla::MyDriver& driver) {
        driver.tpdo_mapped[0x6040][0] = static_cast<uint16_t>(0x02); // Mantém a tensão ativa
        auto time_s = static_cast<double>(t.current_us - initial_time) / 1000000.f;

        calculatedSpeed = (mi.currentAngle - mi.prevAngle) / (static_cast<float>(t.current_dt_us) / 1000000.f);

        // Calcula o erro de posição e de velocidade, e integra o erro para o controle I
        errorPosition = referencePosition - mi.currentAngle;
        errorVelocity = referenceVelocity - calculatedSpeed;
        errorPositionSum += errorPosition * t.current_dt_us / 1000000.f;

        // Calcula o sinal de controle com os ganhos P, I, e D
        controlActionP = kp * errorPosition;
        controlActionD = kd * errorVelocity;
        controlActionI = ki * errorPositionSum;
        controlSignal = controlActionP + controlActionD + controlActionI;

        driver.tpdo_mapped[0x2030][0] = static_cast<int16_t>(controlSignal);

        // Registra o dado no log se o driver estiver pronto
        if(driver.IsReady()) {
            log.addDataPoint(DataPoint{time_s * 1000000, mi.currentPulses, controlSignal, mi.currentCurrent, mi.currentRotationUnfiltered, calculatedSpeed, 0, 0, 0});
             //'const std::tuple<unsigned long long, int, double, short, int, float>';
        }
    };

    // Define variáveis de estado e referência para o controlador DLQR
arma::colvec x{0, 0};                   // Estado inicial {posição, velocidade}
arma::mat controlInput;                  // Entrada de controle calculada
arma::colvec error_ref{x};               // Erro em relação à referência
arma::colvec x_ref{x};                   // Estado de referência
auto reference = [&](double time_s)->double { return 400 * std::cos(2 * M_PI * time_s); }; // Função que calcula a referência com uma senoide
double max_error{0};                     // Erro máximo permitido
double error{0};                         // Erro atual

// Callback para o modo de controle DLQR
manopla::SyncCallback onSyncCallbackDLQR = [&](const manopla::Time& t, const manopla::MotorInfo& mi, manopla::MyDriver& driver) {
    driver.tpdo_mapped[0x6040][0] = static_cast<uint16_t>(0x02); // Ativa a tensão
    
    // Calcula o tempo atual e a velocidade a partir dos dados do motor
    auto time_us = t.current_us - initial_time;
    auto time_s = static_cast<double>(time_us) / 1000000.f;
    calculatedSpeed = (mi.currentAngle - mi.prevAngle) / (static_cast<float>(t.current_dt_us) / 1000000.f);

    // Atualiza o estado x com a posição e velocidade atuais
    x.at(0) = mi.currentPulses;
    x.at(1) = calculatedSpeed;

    // Calcula a referência e o erro, dependendo do tipo de referência (senoide ou constante zero)
    if (referenceType == ReferenceType::SineWave) {
        x_ref = {reference(time_s), 0};
        error_ref = -(x_ref - x);
    }

    // Calcula a entrada de controle com base no tipo de referência
    if (referenceType == ReferenceType::ConstantZero) {
        controlInput = (sys_dlqr_K * x);
    } else if (referenceType == ReferenceType::SineWave) {
        controlInput = (sys_dlqr_K * error_ref);
    }

    // Define o sinal de controle com o valor calculado
    controlSignal = controlInput.at(0);
    driver.tpdo_mapped[0x2030][0] = static_cast<int16_t>(controlSignal);

    // Exibe os dados a cada período determinado
    dt_sum_for_printing_ms += t.current_dt_us / 1000.f;
    if (dt_sum_for_printing_ms > good_print_period_ms) {
        std::cout << "t.current_us: " << setw(10) << t.current_us
                  << " currentAngle:" << setw(10) << mi.currentAngle
                  << " controlInput: " << setw(10) << controlInput.at(0) << '\n';
        dt_sum_for_printing_ms = 0ull;
    }

    // Registra dados no log
    if(driver.IsReady()) {
        if (referenceType == ReferenceType::ConstantZero) {
            log.addDataPoint(DataPoint{time_us, mi.currentPulses, controlSignal, mi.currentCurrent,
                                       mi.currentRotationUnfiltered, calculatedSpeed, 0, 0, 0});
        } else if (referenceType == ReferenceType::SineWave) {
            log.addDataPoint(DataPoint{time_us, mi.currentPulses, controlSignal, mi.currentCurrent,
                                       mi.currentRotationUnfiltered, calculatedSpeed, x_ref.at(0), 0, error_ref.at(0)});
        }
    }
};

// Inicializa variáveis de estado para o modo DLQR-Event
x = {0, 0};
arma::colvec x_checkpoint{x};
double E{0};
uint refresh_count{0};
uint time_count{0};
std::vector<uint64_t> vec_refreshed{};
controlInput = arma::zeros(1, 2) * arma::zeros(2, 1);

// Callback para o modo de controle DLQR-Event
manopla::SyncCallback onSyncCallbackDLQREvent = [&](const manopla::Time& t, const manopla::MotorInfo& mi, manopla::MyDriver& driver) {
    driver.tpdo_mapped[0x6040][0] = static_cast<uint16_t>(0x02);

    // Calcula o tempo e velocidade atuais
    auto time_us = t.current_us - initial_time;
    auto time_s = static_cast<double>(time_us) / 1000000.f;
    calculatedSpeed = (mi.currentAngle - mi.prevAngle) / (static_cast<float>(t.current_dt_us) / 1000000.f);

    x.at(0) = mi.currentPulses;
    x.at(1) = calculatedSpeed;

    // Calcula o erro e a referência
    if (referenceType == ReferenceType::SineWave) {
        x_ref = {reference(time_s), 0};
        error_ref = -(x_ref - x);
    }

    // Calcula o erro máximo permitido com base em uma função `super_event`
    if (referenceType == ReferenceType::ConstantZero) {
        max_error = super_event(arma::norm(x, 2), matrix_norm, time_s, event_a, event_b, event_sigma);
    } else {
        max_error = super_event(arma::norm(error_ref, 2), matrix_norm, time_s, event_a, event_b, event_sigma);
    }

    // Verifica se o erro ultrapassa o erro máximo e, se necessário, atualiza o estado e a entrada de controle
    if (error > max_error) {
        if (referenceType == ReferenceType::ConstantZero) {
            controlInput = (sys_dlqr_K * x);
            x_checkpoint = x;
        } else if (referenceType == ReferenceType::SineWave) {
            controlInput = (sys_dlqr_K * error_ref);
            x_checkpoint = error_ref;
        }
        refresh_count++;
        vec_refreshed.push_back(time_us);
    }
    time_count++;

    controlSignal = controlInput.at(0);

    // Calcula o erro atual
    if (referenceType == ReferenceType::ConstantZero) {
        error = arma::norm(x_checkpoint - x, 2);
    } else if (referenceType == ReferenceType::SineWave) {
        error = arma::norm(x_checkpoint - error_ref, 2);
    }

    driver.tpdo_mapped[0x2030][0] = static_cast<int16_t>(controlSignal);

    // Exibe dados periodicamente
    dt_sum_for_printing_ms += t.current_dt_us / 1000.f;
    if (dt_sum_for_printing_ms > good_print_period_ms) {
        std::cout << "t.current_us: " << setw(10) << t.current_us
                  << " time_s: " << setw(10) << setprecision(2) << time_s
                  << " calculatedSpeed: " << setw(10) << calculatedSpeed
                  << " currentAngle:" << setw(10) << mi.currentAngle
                  << " error:" << setw(10) << error
                  << " max_error:" << setw(10) << max_error
                  << " controlInput: " << setw(10) << controlInput.at(0)
                  << " x_ref: " << setw(10) << x_ref << "\n";
        dt_sum_for_printing_ms = 0ull;
    }

    // Registra dados no log
    if(driver.IsReady()) {
        if (referenceType == ReferenceType::ConstantZero) {
            log.addDataPoint(DataPoint{time_us, mi.currentPulses, controlSignal, mi.currentCurrent,
                                       mi.currentRotationUnfiltered, calculatedSpeed, 0, max_error, error});
        } else if (referenceType == ReferenceType::SineWave) {
            log.addDataPoint(DataPoint{time_us, mi.currentPulses, controlSignal, mi.currentCurrent,
                                       mi.currentRotationUnfiltered, calculatedSpeed, x_ref.at(0), max_error, error});
        }
    }
};

// Configura o controlador no modo desejado
if (synccallback == "position-mode") {
    manopla.installOnSyncCallback(onSyncCallbackPositionMode);
} else if (synccallback == "current-mode") {
    manopla.installOnSyncCallback(onSyncCallbackCurrentMode);
} else if (synccallback == "pid") {
    manopla.installOnSyncCallback(onSyncCallbackPID);
} else if (synccallback == "dlqr") {
    manopla.installOnSyncCallback(onSyncCallbackDLQR);
} else if (synccallback == "dlqr-event") {
    manopla.installOnSyncCallback(onSyncCallbackDLQREvent);
}

// Inicia o loop do controlador e exibe a taxa de atualização
manopla.start_loop();
float refresh_rate = (synccallback == "dlqr-event") ? (100.f * refresh_count) / time_count : 100.f;
std::cout << "\n\nREFRESH RATE: " << refresh_rate << " (" << refresh_count << " / " << time_count << ")\n\n";

// Salva os dados registrados no log
log.saveToFile(ss.str(), synccallback, referenceType, sys_matrix_A, sys_matrix_B, sys_dlqr_K, event_a, event_b, event_sigma, refresh_rate);

return 0;

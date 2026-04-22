import numpy as np


# 초기 해 생성 함수
def generate_initial_population(N):
    initial_population = np.random.randint(0, 2, (N, 2, 5))
    return initial_population


# 적합도 평가 함수
def evaluate_fitness(solution):
    x1 = sum([2 ** (4 - i) * solution[0][i] for i in range(5)])
    x2 = sum([2 ** (4 - i) * solution[1][i] for i in range(5)])

    constraint_1 = (100 * x1 + 50 * x2) <= 3000
    constraint_2 = (10 * x1) <= 100

    if constraint_1 and constraint_2:
        fitness_value = 100 * x1 + 40 * x2
    else:
        fitness_value = 0  # 제약을 하나라도 위반하면 적합도 0점

    return fitness_value


# 교차 연산 함수
def crossover(solution1, solution2):
    crossover_point_1 = np.random.randint(1, 4)  # x1에 대한 교차 지점
    crossover_point_2 = np.random.randint(1, 4)  # x2에 대한 교차 지점

    child = np.empty((2, 5), dtype=int)  # 자식을 빈 배열로 생성

    # 부모 유전자 가져오기
    child[0][:crossover_point_1] = solution1[0][:crossover_point_1]
    child[0][crossover_point_1:] = solution2[0][crossover_point_1:]

    child[1][:crossover_point_2] = solution1[1][:crossover_point_2]
    child[1][crossover_point_2:] = solution2[1][crossover_point_2:]

    return child


# 돌연변이 연산 함수
def mutation(child, p):
    for row in range(2):
        for col in range(5):
            if np.random.random() < p:
                child[row, col] = 1 - child[row, col]

    return child


num_iter = 10               # 세대 수
N = 20                      # 한 세대에 포함되는 해의 개수
N_P = 10                    # 부모 개수
mutation_sol_prob = 0.1     # 해가 돌연변이 연산을 수행할 확률
mutation_gene_prob = 0.2    # 각 유전자가 돌연변이될 확률

# 초기 해 생성
current_population = generate_initial_population(N)
best_score = -1  # 지금까지 찾은 최대 적합도 초기화
best_solution = None

for _ in range(num_iter - 1):
    # 해 평가 수행
    fitness_value_list = np.array(
        [evaluate_fitness(solution) for solution in current_population]
    )

    # 지금까지 찾은 최대 적합도보다 현세대에 있는 최대 적합도가 크다면 업데이트
    if fitness_value_list.max() > best_score:
        best_score = fitness_value_list.max()
        best_solution = current_population[fitness_value_list.argmax()]

    # 적합도 기준 상위 N_P개 해 선정
    parents = current_population[np.argsort(-fitness_value_list)[:N_P]]

    # 새로운 해 집단 정의
    new_population = parents.copy()

    # 두 개의 부모를 선택하면서 자식 생성
    for _ in range(N - N_P):
        # 부모 선택
        parent_1_idx, parent_2_idx = np.random.choice(N_P, 2, replace=False)
        parent_1 = parents[parent_1_idx]
        parent_2 = parents[parent_2_idx]

        # 자식 생성
        child = crossover(parent_1, parent_2)

        # mutation_sol_prob의 확률로 돌연변이 연산 수행
        if np.random.random() < mutation_sol_prob:
            child = mutation(child, mutation_gene_prob)

        # new_population에 child 추가
        new_population = np.vstack([new_population, child.reshape(1, 2, 5)])

    # 다음 세대로 갱신
    current_population = new_population


print("best_score :", best_score)
print("best_solution")
print(best_solution)
"""Mặc định an toàn của API: CORS đóng, lệnh ghi cần token khi được đặt, cổng chỉ ở loopback.

Trước đây CORS là "*" và không có xác thực nào, nên bất kỳ trang web nào người dùng
đang mở cũng gọi được /sync, /fit hay ?record=true vào API chạy trên máy họ. Với
?record=true và POST thân rỗng, CORS chặt cũng không đủ: trình duyệt vẫn GỬI một
"simple request" dù không cho đọc phản hồi. Đầu mục Authorization buộc preflight.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pytest
import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vietlott_engine.api.main import create_app
from vietlott_engine.api.routers import analytics
from vietlott_engine.core.config import Settings
from vietlott_engine.core.games import MEGA_645
from vietlott_engine.core.models import Draw
from vietlott_engine.crawler.storage import InMemoryRepository
from vietlott_engine.ml_models.gcn import SkillReport
from tests.conftest import make_history

TOKEN = "s3cret-token-for-tests"
EVIL = "https://evil.example"


def _repo() -> InMemoryRepository:
    repo = InMemoryRepository()
    h = make_history(MEGA_645, 120, seed=5)
    repo.upsert(
        Draw(game=MEGA_645.code, draw_id=i + 1, draw_date=str(h.dates[i]), numbers=tuple(int(x) for x in h.numbers[i]))
        for i in range(len(h))
    )
    return repo


@pytest.fixture()
def guarded(tmp_path: Path) -> TestClient:
    settings = Settings(storage_backend="memory", api_token=TOKEN, data_dir=tmp_path,
                        forecast_dir=tmp_path / "forecast", calibration_dir=tmp_path / "calibration",
                        auto_update_dir=tmp_path / "updates")
    with TestClient(create_app(settings, repository=_repo())) as c:
        yield c


@pytest.fixture()
def default(tmp_path: Path) -> TestClient:
    settings = Settings(storage_backend="memory", data_dir=tmp_path,
                        forecast_dir=tmp_path / "forecast", calibration_dir=tmp_path / "calibration",
                        auto_update_dir=tmp_path / "updates")
    with TestClient(create_app(settings, repository=_repo())) as c:
        yield c


EV = ("/games/mega645/ev", {"ticket": [3, 17, 22, 35, 41, 44], "jackpot1": 45e9})
GCN_PATHS = ("/games/mega645/ml/gcn/next", "/games/mega645/ml/gcn/skill")


@pytest.mark.parametrize("client_fixture", ["guarded", "default"])
def test_security_clients_keep_forecast_artifacts_in_temporary_data(
    request: pytest.FixtureRequest, tmp_path: Path, client_fixture: str,
) -> None:
    client = request.getfixturevalue(client_fixture)
    assert client.app.state.vqe.settings.data_dir.resolve() == tmp_path
    response = client.get("/ml/forecast/mega645", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert (tmp_path / "forecast" / "ml" / "mega645.json.gz").is_file()


@pytest.fixture()
def gcn_training(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Giữ router và middleware thật; thay phần huấn luyện tốn CPU bằng kết quả cố định."""
    calls: list[str] = []

    def fit(self, *_):
        calls.append("next")
        return self

    def skill(*_, **__):
        calls.append("skill")
        return SkillReport(**{
            name: "gcn" if name == "model" else 20 if name == "test_draws" else 0.0
            for name in SkillReport.model_fields
        })

    monkeypatch.setattr(analytics.GCNPredictor, "fit", fit)
    monkeypatch.setattr(analytics.GCNPredictor, "predict_proba", lambda *_: np.full((1, 45), 6 / 45))
    monkeypatch.setattr(analytics, "walk_forward_skill", skill)
    return calls


@pytest.mark.parametrize("path", GCN_PATHS)
def test_training_get_requires_the_configured_token(
    guarded: TestClient, gcn_training: list[str], path: str,
) -> None:
    assert guarded.get(path).status_code == 401
    assert guarded.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert not gcn_training, "yêu cầu bị chặn không được chạy huấn luyện"
    assert guarded.get(path, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert gcn_training == [path.rsplit("/", 1)[-1]]


@pytest.mark.parametrize("path", GCN_PATHS)
@pytest.mark.parametrize("headers", [{"Origin": EVIL}, {"Sec-Fetch-Site": "cross-site"}])
def test_without_a_token_cross_site_training_get_is_refused(
    default: TestClient, gcn_training: list[str], path: str, headers: dict,
) -> None:
    assert default.get(path, headers=headers).status_code == 403
    assert not gcn_training, "yêu cầu từ trang lạ không được chạy huấn luyện"
    assert default.get(path).status_code == 200
    assert gcn_training == [path.rsplit("/", 1)[-1]]


@pytest.mark.parametrize("deployment", ["mounted", "root_path"])
@pytest.mark.parametrize("path", GCN_PATHS)
def test_a_prefixed_training_get_cannot_skip_the_token(
    tmp_path: Path, deployment: str, path: str, gcn_training: list[str],
) -> None:
    app = create_app(Settings(storage_backend="memory", api_token=TOKEN, data_dir=tmp_path,
                              seed_file_dir=None, auto_update_enabled=False), repository=_repo())
    with ExitStack() as stack:
        if deployment == "mounted":
            stack.enter_context(TestClient(app))
            parent = FastAPI()
            parent.mount("/vqe", app)
            client = TestClient(parent)
        else:
            client = TestClient(app, root_path="/vqe")
        stack.enter_context(client)
        # URL có ký tự mã hoá và dấu / cuối vẫn đi qua cùng cổng trước router/redirect.
        target = "/vqe" + path.replace("mega645", "%6Dega645") + "/?record=off"
        assert client.get(target).status_code == 401
        assert not gcn_training
        assert client.get(target, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert gcn_training == [path.rsplit("/", 1)[-1]]


def test_without_a_configured_token_requests_behave_as_before(default: TestClient) -> None:
    assert default.post(EV[0], json=EV[1]).status_code == 200


def test_with_a_token_writes_need_it_and_reads_do_not(guarded: TestClient) -> None:
    assert guarded.get("/health").status_code == 200
    assert guarded.post(EV[0], json=EV[1]).status_code == 401
    assert guarded.post(EV[0], json=EV[1], headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert guarded.post(EV[0], json=EV[1], headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_openapi_exposes_bearer_for_guarded_try_it_out(guarded: TestClient) -> None:
    """Swagger phải biết gửi token cho POST và GET làm mới trạng thái."""
    schema = guarded.get("/openapi.json").json()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    bearer = next((name for name, value in schemes.items()
                   if value.get("type") == "http" and value.get("scheme") == "bearer"), None)
    assert bearer is not None, "Swagger thiếu HTTP Bearer nên không có nút Authorize"
    for path, method in (
        ("/games/{game}/ev", "post"),
        ("/games/{game}/prizes/sync", "post"),
        ("/forecast/{product}/fit", "post"),
        ("/forecast/{product}", "get"),
        ("/forecast/{product}/evidence", "get"),
        ("/ml/forecast/{product}", "get"),
        ("/games/{game}/ml/gcn/next", "get"),
        ("/games/{game}/ml/gcn/skill", "get"),
    ):
        assert schema["paths"][path][method].get("security") == [{bearer: []}], (path, method)
    for path in ("/health", "/games/{game}/draws", "/forecast/{product}/scoreboard"):
        assert not schema["paths"][path]["get"].get("security"), path
    assert TOKEN not in guarded.get("/openapi.json").text
    # Đây là đầu mục mà client theo HTTP Bearer trong schema phải gửi.
    assert guarded.post(EV[0], json=EV[1]).status_code == 401
    assert guarded.post(EV[0], json=EV[1], headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_openapi_keeps_local_no_token_operations_anonymous(default: TestClient) -> None:
    schema = default.get("/openapi.json").json()
    assert not schema.get("components", {}).get("securitySchemes")
    assert not schema["paths"]["/games/{game}/ev"]["post"].get("security")


def test_record_true_is_a_write_even_though_it_is_a_get(guarded: TestClient) -> None:
    assert guarded.get("/forecast/mega645?record=true").status_code == 401
    assert guarded.get("/forecast/mega645?record=TRUE").status_code == 401


@pytest.mark.parametrize("value", ["on", "ON", "y", "t", "1", "yes"])
def test_every_value_fastapi_reads_as_true_needs_the_token(guarded: TestClient, value: str) -> None:
    # FastAPI đọc tham số bool bằng pydantic: "on", "y", "t"… cũng là True. Middleware phải
    # đọc giống hệt, nếu không ?record=on ghi vào sổ mà không cần token.
    assert guarded.get(f"/forecast/mega645?record={value}").status_code == 401


def test_a_repeated_record_parameter_cannot_hide_a_write(guarded: TestClient) -> None:
    assert guarded.get("/forecast/mega645?record=0&record=on").status_code == 401
    assert guarded.get("/forecast/mega645?record=on&record=0").status_code == 401


def test_an_unreadable_record_value_is_treated_as_a_write(guarded: TestClient) -> None:
    # FastAPI trả 422 cho giá trị này; middleware vẫn đóng an toàn thay vì cho qua không token.
    assert guarded.get("/forecast/mega645?record=maybe").status_code == 401


def test_record_false_and_no_record_stay_reads(guarded: TestClient) -> None:
    # /games/{game}/draws chỉ đọc: record=false (hay không có) không biến nó thành lệnh ghi.
    for query in ("", "?record=false", "?record=off", "?record=0"):
        assert guarded.get(f"/games/mega645/draws{query}").status_code != 401, query


@pytest.mark.parametrize("path", ["/forecast/mega645", "/forecast/mega645/evidence", "/ml/forecast/mega645"])
def test_a_get_that_refreshes_forecast_state_is_a_write(guarded: TestClient, default: TestClient, path: str) -> None:
    # Các GET này học thêm từ kỳ mới, ghi sự kiện vào sổ và lưu checkpoint, dù không có record.
    assert guarded.get(path).status_code == 401
    assert guarded.get(path, headers={"Authorization": f"Bearer {TOKEN}"}).status_code != 401
    assert default.get(path, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_the_scoreboard_only_reads_the_ledger(guarded: TestClient) -> None:
    assert guarded.get("/forecast/mega645/scoreboard").status_code == 200


def test_a_bodyless_cross_site_post_to_sync_is_refused(guarded: TestClient) -> None:
    response = guarded.post("/games/mega645/prizes/sync", headers={"Origin": EVIL})
    assert response.status_code == 401


def test_cors_is_closed_by_default(default: TestClient) -> None:
    response = default.get("/health", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in response.headers


def test_cors_opens_only_to_configured_origins() -> None:
    settings = Settings(storage_backend="memory", cors_origins=["https://ok.example"])
    with TestClient(create_app(settings, repository=_repo())) as c:
        assert c.get("/health", headers={"Origin": "https://ok.example"}).headers.get(
            "access-control-allow-origin") == "https://ok.example"
        assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": EVIL}).headers


def test_docker_publishes_the_api_on_loopback_only() -> None:
    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text(encoding="utf-8"))
    ports = [str(p) for p in compose["services"]["api"]["ports"]]
    assert ports and all(p.startswith("127.0.0.1:") for p in ports), ports


def test_a_rejected_write_still_carries_cors_headers_for_an_allowed_origin() -> None:
    # Xác thực nằm NGOÀI CORS thì 401 trả sớm không có Access-Control-Allow-Origin, và client
    # hợp lệ chỉ thấy lỗi mạng mờ thay vì phản hồi xác thực.
    settings = Settings(storage_backend="memory", api_token=TOKEN, cors_origins=["https://ok.example"])
    with TestClient(create_app(settings, repository=_repo())) as c:
        response = c.post(EV[0], json=EV[1], headers={"Origin": "https://ok.example"})
        assert response.status_code == 401
        assert response.headers.get("access-control-allow-origin") == "https://ok.example"


@pytest.mark.parametrize("headers", [{"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"},
                                     {"Origin": EVIL}])
def test_without_a_token_a_browser_write_from_another_site_is_refused(default: TestClient, headers: dict) -> None:
    # Không đặt token thì client không phải trình duyệt (scheduler, curl) vẫn ghi được, nhưng
    # một trang lạ không còn kích hoạt được lệnh ghi bằng "simple request".
    assert default.get("/forecast/mega645?record=true", headers=headers).status_code == 403
    assert default.post(EV[0], json=EV[1], headers=headers).status_code == 403


def test_without_a_token_same_origin_and_non_browser_writes_still_work(default: TestClient) -> None:
    assert default.post(EV[0], json=EV[1]).status_code == 200
    assert default.post(EV[0], json=EV[1], headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert default.post(EV[0], json=EV[1], headers={"Sec-Fetch-Site": "none"}).status_code == 200


def test_without_a_token_the_api_s_own_docs_page_can_still_write(default: TestClient) -> None:
    # Swagger UI ở /docs do chính API phục vụ: trình duyệt gửi Origin là host của API.
    own = {"Origin": "http://testserver"}
    assert default.post(EV[0], json=EV[1], headers={**own, "Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert default.post(EV[0], json=EV[1], headers=own).status_code == 200


def test_an_origin_listed_in_cors_may_write_without_a_token() -> None:
    settings = Settings(storage_backend="memory", cors_origins=["https://ok.example"])
    with TestClient(create_app(settings, repository=_repo())) as c:
        headers = {"Origin": "https://ok.example", "Sec-Fetch-Site": "cross-site"}
        assert c.post(EV[0], json=EV[1], headers=headers).status_code == 200


def test_serve_listens_on_loopback_unless_told_otherwise() -> None:
    from vietlott_engine.cli import build_parser

    assert build_parser().parse_args(["serve"]).host == "127.0.0.1"


@pytest.mark.parametrize("deployment", ["mounted", "root_path"])
@pytest.mark.parametrize("path", ["/forecast/mega645", "/forecast/mega645/evidence", "/ml/forecast/mega645"])
def test_a_prefixed_forecast_get_cannot_write_without_the_token(tmp_path: Path, deployment: str, path: str) -> None:
    """Tiền tố ASGI không được làm GET ghi trạng thái lọt qua xác thực."""
    app = create_app(Settings(storage_backend="memory", api_token=TOKEN, data_dir=tmp_path,
                              seed_file_dir=None, auto_update_enabled=False), repository=_repo())
    with ExitStack() as stack:
        if deployment == "mounted":
            # ASGI mount không tự chạy lifespan của app con: khởi động app thật trước.
            stack.enter_context(TestClient(app))
            parent = FastAPI()
            parent.mount("/vqe", app)
            client = TestClient(parent)
        else:
            client = TestClient(app, root_path="/vqe")
        stack.enter_context(client)
        assert client.get("/vqe" + path).status_code == 401
        assert not list(tmp_path.rglob("*.json")), "yêu cầu bị chặn không được tạo checkpoint hay sổ"
        assert client.get("/vqe" + path, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


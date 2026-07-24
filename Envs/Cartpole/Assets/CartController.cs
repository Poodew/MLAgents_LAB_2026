using UnityEngine;
using UnityEngine.UIElements;

public class CartController : MonoBehaviour
{
    [SerializeField] Rigidbody cartRb; //카트의 물리 정보 
    [SerializeField] Rigidbody poleRb; //기둥의 물리 정보
    [SerializeField] HingeJoint pole; //기둥 Hinge(각을 얻기 위해 사용)

    public float cartX; //카트의 위치
    public float cartVel; //카트의 속도
    public float poleAngle; //기둥의 각도
    public float poleAngVel; //기둥의 각속도

    public float currentTime = 0; //에피소드 진행 시간

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Application.targetFrameRate = 60;

        //기둥의 각도를 -5 ~ 5도 사이로 초기화
        poleRb.rotation = Quaternion.Euler(0, 0, Random.Range(-5.0f, 5.0f));
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        UpdateStates();
        if (IsFailed())
            ResetCart();
        Move(Input.GetAxisRaw("Horizontal"));
    }

    //매 프레임마다 상태 업데이트
    public void UpdateStates()
    {
        cartX = cartRb.transform.position.x;
        cartVel = cartRb.linearVelocity.x;
        poleAngle = pole.angle;
        poleAngVel = poleRb.angularVelocity.z;

        currentTime += Time.deltaTime;
    }

    //카트 이동
    public void Move(float input)
    {
        cartRb.AddForce(Vector3.right * input * 2.5f);
    }

    //에피소드 종료 여부 체크
    public bool IsFailed()
    {
        if ((cartX < -1.0f || cartX > 1.0f) || (poleAngle < -80.0f || poleAngle > 80.0f))
            return true;

        return false;
    }

    //에피소드 초기화
    public void ResetCart()
    {
        currentTime = 0;

        cartRb.position =
            new Vector3(Random.Range(-0.2f, 0.2f), 0, 0);
        poleRb.position = Vector3.zero;

        cartRb.rotation = Quaternion.identity;
        poleRb.rotation =
            Quaternion.Euler(0, 0, Random.Range(-5.0f, 5.0f));

        cartRb.linearVelocity = Vector3.zero;
        cartRb.angularVelocity = Vector3.zero;

        poleRb.linearVelocity = Vector3.zero;
        poleRb.angularVelocity = Vector3.zero;
    }
}
